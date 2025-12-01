import os
import json
import torch
import torch.serialization
from lightning_fabric.utilities.data import AttributeDict
import torch.nn as nn
import pytorch_lightning as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
from lightning_tools.optim import config_optimizer
import numpy as np
import pynvml, psutil, time, csv
pynvml.nvmlInit()

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
        # 🔹 Case 1: Low-resource mode → 4-bit + QLoRA
        # ============================================================
        if args.low_resource:
            print("→ Low resource mode detected: loading 4-bit model with QLoRA")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # ⛔ DDP-safe: no device_map="auto"
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=None,   # ❌ jangan "auto" (DDP unsafe)
            )

            # ✅ Tambahkan LoRA (QLoRA)
            print("Applying QLoRA...")
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.llm_lora_dropout,
                bias="none",
                target_modules=["q_proj","v_proj", "o_proj", "k_proj"],

            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("Loading 4-bit QLoRA LLAMA Done ✅")
                    
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
        # Linear projection for visual features → LLAMA space
        # ============================================================
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        # ============================================================
        # Print parameter info untuk Visual Mapper
        # ============================================================
        mapper_params = sum(p.numel() for p in self.llama_proj.parameters()) \
                    + sum(p.numel() for p in self.layer_norm.parameters())

        mapper_trainable = sum(p.numel() for p in self.llama_proj.parameters() if p.requires_grad) \
                        + sum(p.numel() for p in self.layer_norm.parameters() if p.requires_grad)

        print(f"[Visual Mapper] trainable params: {mapper_trainable:,} "
            f"|| all params: {mapper_params:,} "
            f"|| trainable%: {100 * mapper_trainable / mapper_params:.4f}%")

        # ============================================================
        # Print TOTAL PARAMETERS (VE + Mapper + LLM)
        # ============================================================
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())

        print(f"[TOTAL ALIGNMENT] trainable params: {total_trainable:,} "
            f"|| all params: {total_all:,} "
            f"|| trainable%: {100 * total_trainable / total_all:.4f}%")

        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        # ====== buffers for training & validation ======
        self.val_step_outputs = []         # untuk menyimpan hypo/ref per batch (val)
        self._epoch_vram = []              # VRAM real per-step (train/val)
        self._train_epoch_start_time = None
        self._val_epoch_start_time = None
        self.val_score = 0.0               # best val score untuk save checkpoint

        # ====== buffers for testing ======
        self.test_step_outputs = []        # hypo/ref test
        self.test_latencies = []           # latency per batch
        self.test_vrams = []               # VRAM per batch
        self.test_cpu_rams = []            # CPU RAM per batch

        if args.delta_file is not None:
            # Izinkan AttributeDict supaya tidak error saat load
            torch.serialization.add_safe_globals([AttributeDict])

            # ✅ Load checkpoint (PyTorch >= 2.6 perlu weights_only=False)
            state = torch.load(args.delta_file, map_location='cuda', weights_only=False)

            # Ambil state_dict model (fleksibel)
            state_dict = state.get('model', state)

            # Load ke model
            self.load_state_dict(state_dict, strict=False)
            print(f'✅ Loaded checkpoint from {args.delta_file}')
            
    def _log_gpu_cpu_epoch(self, prefix):
        """
        Logging peak VRAM & CPU RAM per epoch.
        prefix = "train" atau "val"
        """

        # Ambil VRAM real per-step (sudah disimpan ke self._epoch_vram)
        peak_vram_local = float(max(self._epoch_vram)) if self._epoch_vram else 0.0

        # All-gather peak vram ke semua GPU
        t_vram = torch.tensor(peak_vram_local, device=self.device)
        all_vram = self.all_gather(t_vram).detach().cpu().tolist()  # [gpu0, gpu1]

        if self.trainer.is_global_zero:
            # VRAM real (GB) — format 2 desimal TANPA ROUND (truncate)
            gpu0_vram = all_vram[0]
            gpu1_vram = all_vram[1]

            self.log(f"{prefix}_gpu0_vram", gpu0_vram, on_step=False, on_epoch=True)
            self.log(f"{prefix}_gpu1_vram", gpu1_vram, on_step=False, on_epoch=True)

            # CPU RAM real (GB)
            mem = psutil.virtual_memory()
            cpu_ram = mem.used / (1024**3)
            self.log(f"{prefix}_cpu_ram", cpu_ram, on_step=False, on_epoch=True)

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
        img_embeds = self.layer_norm(img_embeds)

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
        self._epoch_vram = []
        self._epoch_util = []
        # catat waktu mulai epoch train (dalam detik)
        self._train_epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        result = self(batch)
        # Untuk progress bar — tampil tiap step
        self.log("train_loss_step", result["loss"], on_step=True, on_epoch=False, prog_bar=True,logger=False)
        
        # Untuk logger — mean loss per epoch
        self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # track vram per-step
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        # VRAM real usage
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)

        self._epoch_vram.append(vram)

        return result["loss"]
        
    def on_train_epoch_end(self):
        # hitung waktu epoch train dalam jam
        if self._train_epoch_start_time is not None:
            train_epoch_time_sec = time.time() - self._train_epoch_start_time
            train_epoch_time_hr = train_epoch_time_sec / 3600.0  # konversi ke jam
            self.log("train_epoch_time", train_epoch_time_hr, on_step=False, on_epoch=True, rank_zero_only=True)

        self._log_gpu_cpu_epoch("train")

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
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
        
    def on_validation_epoch_start(self):
        self._epoch_vram = []
        self._epoch_util = []
        self._val_epoch_start_time = time.time()
        
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
        img_embeds = self.layer_norm(img_embeds)
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
        
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)

        self._epoch_vram.append(vram)

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
        
        # 🔄 Sinkron antar GPU (tanpa all_reduce manual)
        for k, v in eval_res.items():
            eval_res[k] = self.trainer.strategy.reduce(v, reduce_op="mean")
            
        # Log semua metrik evaluasi
        self.log_dict(eval_res, sync_dist=False, logger=True)
        
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        # ⬇️ Tambahkan rank di nama file agar tidak tabrakan
        rank = self.global_rank if hasattr(self, "global_rank") else 0
        json.dump(hypo, open(os.path.join(result_folder, f"result_rank{rank}_{current_epoch}_{global_step}.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, f"refs_rank{rank}.json"), "w"))
        self.print(eval_res)

        val_score = 0.0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        # 💾 Simpan checkpoint hanya di rank 0, tapi berdasarkan val_score global
        if self.trainer.is_global_zero:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score

        # 🧹 Bersihkan buffer
        self.val_step_outputs.clear()
        # hitung waktu epoch val dalam jam
        if self._val_epoch_start_time is not None:
            val_epoch_time_sec = time.time() - self._val_epoch_start_time
            val_epoch_time_hr = val_epoch_time_sec / 3600.0
            self.log("val_epoch_time", val_epoch_time_hr, on_step=False, on_epoch=True, rank_zero_only=True)

        # log vram/util per-epoch
        self._log_gpu_cpu_epoch("val")
        
        
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
        img_embeds = self.layer_norm(img_embeds)
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
        # --- timing end ---
        end = time.time()
        latency = end - start

        # ===== VRAM (REAL) =====
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3

        # ===== CPU RAM =====
        cpu_ram = psutil.virtual_memory().used / 1024**3

        # ===== Simpan untuk perhitungan final =====
        self.test_latencies.append(latency)
        self.test_vrams.append(vram)
        self.test_cpu_rams.append(cpu_ram)
        
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        
        # ======== PERFORMANCE LOGGING =========
        avg_lat = float(np.mean(self.test_latencies)) if self.test_latencies else 0.0
        std_lat = float(np.std(self.test_latencies)) if self.test_latencies else 0.0
        throughput = 1.0 / avg_lat if avg_lat > 0 else 0.0

        # VRAM & CPU RAM → gunakan PEAK
        peak_vram = float(max(self.test_vrams)) if self.test_vrams else 0.0
        peak_cpu_ram = float(max(self.test_cpu_rams)) if self.test_cpu_rams else 0.0

        # ==== FORMAT ====
        avg_lat = avg_lat
        std_lat = std_lat
        throughput = throughput
        peak_vram = peak_vram
        peak_cpu_ram = peak_cpu_ram

        # Simpan ke CSV
        header = ["avg_latency", "std_latency", "throughput", "gpu_vram", "cpu_ram"]
        data = [avg_lat, std_lat, throughput, peak_vram, peak_cpu_ram]

        csv_path = os.path.join(self.hparams.savedmodel_path, "test_perf_metrics.csv")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

        print(f"Saved test performance metrics to {csv_path}")
    
    def configure_optimizers(self):

        # ------------------------------
        # Hitung total training steps
        # ------------------------------
        total_steps = self.trainer.estimated_stepping_batches

        # Warmup 5% (opsi aman untuk QLoRA, IU-Xray kecil)
        warmup_steps = int(total_steps * 0.05)
        warmup_steps = min(max(warmup_steps, 10), total_steps - 1)

        # ------------------------------
        # Optimizer & Scheduler
        # ------------------------------
        optimizer, scheduler = config_optimizer(
            parameters=self.parameters(),
            init_lr=self.hparams.learning_rate,
            warmup_steps=warmup_steps,
            max_steps=total_steps,
            name="lr"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
