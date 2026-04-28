#!/usr/bin/env bash

tracka_require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing ${label}: $path" >&2
    exit 1
  fi
}

tracka_require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: $path" >&2
    exit 1
  fi
}

tracka_require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing ${label}: $path" >&2
    exit 1
  fi
}

tracka_require_dataset_layout() {
  local data_root="$1"
  tracka_require_dir "$data_root/train" "dataset train dir"
  tracka_require_dir "$data_root/val" "dataset val dir"
}

tracka_prepare_run_dirs() {
  local tmp_root="$1"
  local output_dir="$2"
  local log_dir="$3"
  mkdir -p "$tmp_root" "$output_dir" "$log_dir"
  export TMPDIR="$tmp_root"
  export TMP="$tmp_root"
  export TEMP="$tmp_root"
}

tracka_init_output_log() {
  local output_log="$1"
  : > "$output_log"
  export TRACKA_META_LOG="$output_log"
}

tracka_print_kv() {
  local prefix="$1"
  local key="$2"
  local value="$3"
  local line="[$prefix] ${key}=${value}"
  if [[ -n "${TRACKA_META_LOG:-}" ]]; then
    echo "$line" | tee -a "$TRACKA_META_LOG"
  else
    echo "$line"
  fi
}

tracka_append_common_args() {
  local -n out="$1"
  local data_root="$2"
  local output_dir="$3"
  local log_dir="$4"
  local epochs="$5"
  local seed="$6"
  local ratio_weight="$7"
  local activation_lr_scale="$8"
  local save_ckpt_flag="$9"
  local stop_after_epoch="${10}"
  local cls_distill_weight="${11}"
  local token_distill_weight="${12}"

  out+=(
    --model deit-s
    --data_set image_folder
    --data_path "$data_root/train"
    --eval_data_path "$data_root/val"
    --nb_classes 2
    --output_dir "$output_dir"
    --log_dir "$log_dir"
    --input_size 224
    --batch_size 32
    --epochs "$epochs"
    --num_workers 4
    --base_rate 0.7
    --ratio_weight "$ratio_weight"
    --lr 3e-5
    --warmup_epochs 0
    --warmup_steps 50
    --clip_grad 1.0
    --device cuda
    --save_ckpt "$save_ckpt_flag"
    --save_ckpt_freq 1
    --save_ckpt_num 2
    --auto_resume false
    --use_amp false
    --mixup 0
    --cutmix 0
    --seed "$seed"
    --lr_scale 1.0
    --groupa_lr_scale 0.1
    --activation_lr_scale "$activation_lr_scale"
    --cls_distill_weight "$cls_distill_weight"
    --token_distill_weight "$token_distill_weight"
    --use_square_gelu true
    --square_activation_mode learnable_quadratic_gelu_init
    --use_approx_attn false
    --use_mask_pruning true
    --debug_nan true
    --patch_embed_bias_init_mode zero
    --freeze_patch_embed_proj true
    --pretrained_fix_step 0
    --stop_after_epoch "$stop_after_epoch"
  )
}

tracka_run_training() {
  local python_bin="$1"
  local train_entry="$2"
  local output_log="$3"
  shift 3
  "$python_bin" "$train_entry" "$@" |& tee -a "$output_log"
}
