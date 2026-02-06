import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
from lightning_tools.optim import config_optimizer
import numpy as np
import csv, time, psutil, pynvml
pynvml.nvmlInit()


# ============================================================================
#  MLP MAPPER
# ----------------------------------------------------------------------------
class CompensatoryMlpMapper(nn.Module):
    def __init__(self, visual_dim, llm_dim, hidden_dim=2048, hidden_dropout=0.1):
        super().__init__()
        
        # Layer 1: Middle-neck (Ekspansi ke 2048)
        # Memberikan kapasitas otak yang cukup tanpa meledakkan parameter.
        self.layer1 = nn.Linear(visual_dim, hidden_dim)
        self.activation = nn.GELU()
        
        # Dropout: Satpam Anti-Hafalan (PENTING!)
        self.dropout = nn.Dropout(hidden_dropout)
        
        # Layer 2: Proyeksi Akhir ke Llama
        self.layer2 = nn.Linear(hidden_dim, llm_dim)
        
        # Normalisasi: Wajib untuk kestabilan FP16
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        # Jalur Lurus (Tanpa Jalan Tol/Residual)
        # Memaksa model memproses setiap gambar dengan serius.
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.norm(x)
        return x

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.vis_lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
            
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            trainable_params = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.visual_encoder.parameters())
            print(f"[Vision Encoder] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            trainable_params = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.visual_encoder.parameters())
            print(f"[Vision Encoder] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print(f'Loading Full Trainable vision encoder:{args.vision_model} -- Done')
                
        print('Loading LLAMA model...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0

        # ============================================================
        # 🔹 Low-resource mode: 4-bit LLaMA
        # ============================================================
        if args.low_resource:
            print("→ Low-resource mode detected: loading 4bit model")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_type=torch.float16
            )
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                quantization_config=bnb_config,
                device_map=None  # DDP-safe
            )

            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                
            trainable_params = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llama_model.parameters())
            print(f"[LLM] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print("Loading 8bit LLAMA Done ✅")
        # ============================================================
        # 🔹 Case 2: Full mode → FP16 (no quantization, no LoRA)
        # ============================================================
        else:
            print("→ Full precision mode detected: loading FP16 model")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                device_map=None  # DDP-safe
            )

            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                
            trainable_params = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llama_model.parameters())
            print(f"[LLM] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print("Loading FP16 LLAMA Done ✅")
        
        # ============================================================
        # MAPPER
        # ============================================================
        self.llama_proj = CompensatoryMlpMapper(
            visual_dim=self.visual_encoder.num_features, 
            llm_dim=self.llama_model.config.hidden_size,
            hidden_dim=2048,    # Ukuran pas
            hidden_dropout=0.1  # Cegah overfit
        )
        # ============================================================
        # Print parameter info untuk Visual Mapper
        # ============================================================
        mapper_params = sum(p.numel() for p in self.llama_proj.parameters())
        mapper_trainable = sum(p.numel() for p in self.llama_proj.parameters() if p.requires_grad)
        print(f"[Visual Mapper] trainable params: {mapper_trainable:,} "f"|| all params: {mapper_params:,} "f"|| trainable%: {100 * mapper_trainable / mapper_params:.4f}%")
        
        # ============================================================
        # Print TOTAL parameter info untuk seluruh model
        # ============================================================
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())
        print(f"[TOTAL ALIGNMENT] trainable params: {total_trainable:,} "f"|| all params: {total_all:,} "f"|| trainable%: {100 * total_trainable / total_all:.4f}%")
        
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        # ====== buffers for training & validation ======
        self.val_step_outputs = []         # untuk menyimpan hypo/ref per batch (val)
        self.val_score = 0.0               # best val score untuk save checkpoint
        self.test_step_outputs = []        # hypo/ref test
        self.test_profile = []

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'), weights_only=False)['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
            
    
    def _get_gpu_mem(self):
        gpus = []
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append(round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**3, 2))
        return gpus

    def _get_cpu_mem(self):
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2)

    def _append_system_csv(self, epoch, phase, gpu_mem, cpu_mem, time_h):
        path = os.path.join(self.hparams.savedmodel_path, "profiling")
        os.makedirs(path, exist_ok=True)
        csv_path = os.path.join(path, "train_val_system_profile.csv")

        gpu_cols = [f"gpu{i}_gb" for i in range(len(gpu_mem))]
        header = ["epoch", "phase"] + gpu_cols + ["cpu_gb", "time_h"]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([epoch, phase, *gpu_mem, cpu_mem, time_h])

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    
    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}
    
    def on_train_epoch_start(self):
        torch.cuda.synchronize()
        self._train_start = time.time()

    
    def training_step(self, batch, batch_idx):
        result = self(batch)
        loss = result["loss"]

        # Untuk  progbar — tampil tiap step
        self.log("train_loss_step", result["loss"], on_step=True, on_epoch=False, prog_bar=True,logger=False)
        
        # Untuk logger dan progbar — mean loss per epoch
        self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def on_train_epoch_end(self):
        torch.cuda.synchronize()
        t = round((time.time() - self._train_start) / 3600, 2)
        gpu = self._get_gpu_mem()
        cpu = self._get_cpu_mem()

        if self.trainer.is_global_zero:
            self._append_system_csv(epoch=self.current_epoch, phase="train", gpu_mem=gpu, cpu_mem=cpu, time_h=t)
            
    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'weights'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'weights',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        print("\nSaving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
        
    def on_validation_epoch_start(self):
        torch.cuda.synchronize()
        self._val_start = time.time()
        
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        
        # sync metric ke global (DDP)
        eval_res_synced = {}
        for k, v in eval_res.items():
            t = torch.tensor(v, device=self.device, dtype=torch.float32)
            t_all = self.all_gather(t).mean()
            eval_res_synced[k] = t_all
            
        # Log metric yang SUDAH tersinkron
        self.log_dict(eval_res_synced, logger=True)
                
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        # ⬇️ Tambahkan rank di nama file agar tidak tabrakan
        rank = self.global_rank if hasattr(self, "global_rank") else 0
        json.dump(hypo, open(os.path.join(result_folder, f"result_rank{rank}_{current_epoch}_{global_step}.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, f"refs_rank{rank}.json"), "w"))

        val_score = 0.0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res_synced[score_type] * weight

        # 💾 Simpan checkpoint hanya di rank 0, tapi berdasarkan val_score global
        if self.trainer.is_global_zero:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res_synced)
                self.val_score = val_score
                
        t = round((time.time() - self._val_start) / 3600, 2)
        gpu = self._get_gpu_mem()
        cpu = self._get_cpu_mem()
        
        if self.trainer.is_global_zero:
            self._append_system_csv(epoch=self.current_epoch, phase="val", gpu_mem=gpu, cpu_mem=cpu, time_h=t)       
        
            printable = {k: v.item() for k, v in eval_res_synced.items()}
            print(f"\n[Metric Epoch {self.current_epoch}] {printable}")

        # 🧹 Bersihkan buffer
        self.val_step_outputs.clear()
        
        
    def on_test_epoch_start(self):
        torch.cuda.synchronize()
        self._test_start = time.time()
        self.total_test_samples = 0
    
    def test_step(self, samples, batch_idx):
        start = time.time()
        self.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature, 
        )

        torch.cuda.synchronize()
        latency = time.time() - start
        self.test_profile.append(latency)
        self.total_test_samples += len(samples["id"])

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        return hypo, ref

    def on_test_epoch_end(self):

        # ============================
        #   Kumpulkan hasil hypo/ref
        # ============================
        ref, hypo, ids = [], [], []
        for out in self.test_step_outputs:
            ref.extend(out["ref"])
            hypo.extend(out["hypo"])
            ids.extend(out["id"])

        # Format ulang untuk evaluator
        ref_dict = {k: [v] for k, v in zip(ids, ref)}
        hypo_dict = {k: [v] for k, v in zip(ids, hypo)}

        # ============================
        #   Hitung skor evaluasi
        # ============================
        eval_res = self.score(ref_dict, hypo_dict)

        print(f"\n[TEST RESULT] {eval_res}")

        # ============================
        # Simpan JSON
        # ============================
        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)

        json.dump(hypo_dict, open(
            os.path.join(result_folder, "test_result.json"), "w"))
        json.dump(ref_dict, open(
            os.path.join(result_folder, "test_refs.json"), "w"))

        # ============================
        # Hitung latency & throughput
        # ============================
        total_time_s = time.time() - self._test_start

        # latency rata-rata per sample (seconds/sample)
        avg_sample_latency_s = total_time_s / self.total_test_samples

        # Throughput = samples / second
        throughput = self.total_test_samples / total_time_s
        
        # ============================
        # Hitung TOTAL waktu test set
        # ============================
        torch.cuda.synchronize()
        time_h = round((time.time() - self._test_start) / 3600, 4)

        # ============================
        # Simpan CSV (METRIK + PERF)
        # ============================
        path = os.path.join(self.hparams.savedmodel_path, "profiling")
        os.makedirs(path, exist_ok=True)
        csv_path = os.path.join(path, "test_system_profile.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
                "ROUGE_L", "METEOR", "CIDEr",
                "avg_latency_s",
                "throughput_samples_per_s",
                "time_h"
            ])
            writer.writerow([
                eval_res["Bleu_1"],
                eval_res["Bleu_2"],
                eval_res["Bleu_3"],
                eval_res["Bleu_4"],
                eval_res["ROUGE_L"],
                eval_res["METEOR"],
                eval_res["CIDEr"],
                round(avg_sample_latency_s, 4),
                round(throughput, 4),
                time_h
            ])
        print(f"[OK] Test metrics saved → {csv_path}")
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
