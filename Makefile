SHELL := /bin/bash
# Makefile (place at repo root)
# ==========================================
# 1. Global Configuration
# ==========================================
PYTHON        ?= python
DEVICE        ?= cuda
SEED          ?= 1234
PYWARN        ?= PYTHONWARNINGS="ignore::FutureWarning"
TRAIN_ENTRY   := -m src.cli.train
VIS_ENTRY     := -m src.cli.eval_and_vis
TSPLIB_ENTRY  := -m src.cli.evaluate_tsplib
BENCH_EVAL_ENTRY := -m src.cli.evaluate
THIRD_PARTY_DIR ?= third_party/baselines
TSPLIB_DIR    ?= benchmarks/tsplib

# Problem sizes and data volume
SIZES         ?= 20 50
TRAIN_SAMPLES ?= 50000
ENSURE_UNIQUE ?= 1
GRID_SIZE     ?= 10000

# Evaluation Graph Size (Default for eval_lkh target only)
EVAL_LKH_N    ?= 10000
BENCHMARK     ?= synthetic
SYNTHETIC_N   ?= $(EVAL_LKH_N)

# Architecture / Pruning parameters
R             ?= 4
MAX_POINTS_PER_LEAF ?= 4
MAX_DEPTH           ?= 20

# Training Hyperparameters
TRAIN_N       ?= 50
BATCH_SIZE    ?= 8
EPOCHS        ?= 20
LR            ?= 1e-3
TWO_OPT_PASSES ?= 30
W_BC          ?= 1.0
NUM_WORKERS   ?= 16
NUM_CHECKPOINTS ?= 10
DECODE_BACKEND ?= greedy
EXACT_TIME_LIMIT ?= 30.0
EXACT_LENGTH_WEIGHT ?= 0.0
STATE_MODE ?= iface
MATCHING_MAX_USED ?= 4
RUN_EXACT ?= 0
SETTINGS ?=
# Set to 1 to use LKH-3 for ground truth instead of 2-opt
USE_LKH       ?= 1

# Paths and Logging
DATA_DIR      ?= data
OUT_DIR       ?= outputs
CKPT_DIR      ?= checkpoints
LKH_TAG       := $(if $(filter 1,$(USE_LKH)),_lkh,)
LOG_FILE      ?= $(CKPT_DIR)/train_N$(TRAIN_N)_R$(R)$(LKH_TAG)_p$(TWO_OPT_PASSES).log
LOG_INTERVAL  ?= 200
VAL_INTERVAL  ?= 2000
SAVE_INTERVAL ?= 2000
# Path to the LKH-3 executable (download and compile from http://vectors.uoa.gr/lkh/)
LKH_EXE       ?= data/lkh/LKH-3.0.13/LKH
DATA_STD_DIR  ?= $(DATA_DIR)/std

# Default values for baselines checkpoints (Best -> Latest -> Default fallback)
NEUROLKH_CKPT  ?= $(shell if [ -f checkpoints/neurolkh_baselines/best_ckpt.pt ]; then echo checkpoints/neurolkh_baselines/best_ckpt.pt; else ls -v checkpoints/neurolkh_baselines/epoch_*.pt 2>/dev/null | tail -n 1 || echo checkpoints/neurolkh_baselines/checkpoint.pt; fi)
POMO_LOG       ?= checkpoints/pomo_baselines/train_pomo.log
NEUROLKH_LOG   ?= checkpoints/neurolkh_baselines/train_neurolkh.log
POMO_CKPT      ?= $(shell if [ -f checkpoints/pomo_baselines/best_ckpt.pt ]; then echo checkpoints/pomo_baselines/best_ckpt.pt; else ls -v checkpoints/pomo_baselines/checkpoint-*.pt 2>/dev/null | tail -n 1 || echo checkpoints/pomo_baselines/checkpoint.pt; fi)
TSPLIB_SAVE_DIR ?= $(OUT_DIR)/eval_tsplib
TSPLIB_RUN_TAG ?=
TSPLIB_SET    ?=
TSPLIB_INSTANCES ?=
DATA_PT       ?=
EVAL_OUTPUT_DIR ?=
RUN_TAG       ?=

# Helpers for "20 50" -> "20,50"
empty :=
space := $(empty) $(empty)
comma := ,
SIZES_CSV := $(subst $(space),$(comma),$(strip $(SIZES)))

# ==========================================
# 2. Main Targets
# ==========================================
.PHONY: all deps data spanner raw rlight vis clean train eval vis_pred labels compile_neurolkh train_neuraLKH bench_eval

all: data spanner raw rlight vis

# Install python deps
deps:
	$(PYWARN) $(PYTHON) -m pip install --upgrade pip
	$(PYWARN) $(PYTHON) -m pip install scipy scikit-learn matplotlib tqdm pytz
	$(PYWARN) $(PYTHON) -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
	$(PYWARN) $(PYTHON) -m pip install torch-geometric

# ==========================================
# 3. Data Pipeline
# ==========================================

# Generate points datasets
data:
	@echo "[DATA] Generating points for sizes: $(SIZES)"
	@if [ "$(ENSURE_UNIQUE)" = "1" ]; then \
		$(PYWARN) $(PYTHON) src/utils/data_generator.py \
			--data_dir $(DATA_DIR) \
			--sizes "$(SIZES_CSV)" \
			--num_samples $(TRAIN_SAMPLES) \
			--grid_size $(GRID_SIZE) \
			--seed $(SEED) ; \
	else \
		$(PYWARN) $(PYTHON) src/utils/data_generator.py \
			--data_dir $(DATA_DIR) \
			--sizes "$(SIZES_CSV)" \
			--num_samples $(TRAIN_SAMPLES) \
			--grid_size $(GRID_SIZE) \
			--seed $(SEED) \
			--no_unique ; \
	fi

# Build spanner graphs
spanner: $(foreach N,$(SIZES),\
	$(DATA_DIR)/N$(N)/train_spanner.pt \
	$(DATA_DIR)/N$(N)/val_spanner.pt \
	$(DATA_DIR)/N$(N)/test_spanner.pt)

$(DATA_DIR)/N%/train_spanner.pt: $(DATA_DIR)/N%/train.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@ --num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/val_spanner.pt: $(DATA_DIR)/N%/val.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@ --num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/test_spanner.pt: $(DATA_DIR)/N%/test.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@ --num_workers $(NUM_WORKERS)

# Build raw quadtree pyramid (dependency on Makefile ensures rebuild if params change)
raw: $(foreach N,$(SIZES),\
	$(DATA_DIR)/N$(N)/train_raw_pyramid.pt \
	$(DATA_DIR)/N$(N)/val_raw_pyramid.pt \
	$(DATA_DIR)/N$(N)/test_raw_pyramid.pt)

$(DATA_DIR)/N%/train_raw_pyramid.pt: $(DATA_DIR)/N%/train_spanner.pt

	@echo "[RAW] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py \
		--input $< \
		--output $@ \
		--max_points $(MAX_POINTS_PER_LEAF) \
		--max_depth $(MAX_DEPTH) \
		--num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/val_raw_pyramid.pt: $(DATA_DIR)/N%/val_spanner.pt

	@echo "[RAW] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py \
		--input $< \
		--output $@ \
		--max_points $(MAX_POINTS_PER_LEAF) \
		--max_depth $(MAX_DEPTH) \
		--num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/test_raw_pyramid.pt: $(DATA_DIR)/N%/test_spanner.pt

	@echo "[RAW] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py \
		--input $< \
		--output $@ \
		--max_points $(MAX_POINTS_PER_LEAF) \
		--max_depth $(MAX_DEPTH) \
		--num_workers $(NUM_WORKERS)

# Prune to r-light pyramid (depends on Makefile to auto-rebuild if R changes)
rlight: $(foreach N,$(SIZES),\
	$(DATA_DIR)/N$(N)/train_r_light_pyramid.pt \
	$(DATA_DIR)/N$(N)/val_r_light_pyramid.pt \
	$(DATA_DIR)/N$(N)/test_r_light_pyramid.pt)

$(DATA_DIR)/N%/train_r_light_pyramid.pt: $(DATA_DIR)/N%/train_raw_pyramid.pt

	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R) --num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/val_r_light_pyramid.pt: $(DATA_DIR)/N%/val_raw_pyramid.pt

	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R) --num_workers $(NUM_WORKERS)

$(DATA_DIR)/N%/test_r_light_pyramid.pt: $(DATA_DIR)/N%/test_raw_pyramid.pt

	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R) --num_workers $(NUM_WORKERS)

# Visualize one sample
vis: $(foreach N,$(SIZES),$(OUT_DIR)/vis_N$(N)_train.png)

$(OUT_DIR)/vis_N%_train.png: $(DATA_DIR)/N%/train_r_light_pyramid.pt
	@mkdir -p $(OUT_DIR)
	@echo "[VIS] $< -> $@"
	$(PYWARN) $(PYTHON) src/visualization/visualize_pyramid.py \
		--input $< \
		--sample_idx 0 \
		--output $@ \
		--max_depth_draw 20

# ==========================================
# 4. Training
# ==========================================
TRAIN_DATA := $(DATA_DIR)/N$(TRAIN_N)/train_r_light_pyramid.pt
VAL_DATA   := $(DATA_DIR)/N$(TRAIN_N)/val_r_light_pyramid.pt
# CKPT will be 'None' for train, but 'ckpt_best.pt' for eval/vis (see below)

# continue training:  		make train CKPT="checkpoints/ckpt_best.pt"

# train depends on the pruned data pyramid; if missing, triggers full pipeline for TRAIN_N
train: CKPT := None
train: $(TRAIN_DATA) $(VAL_DATA)
	@mkdir -p $(CKPT_DIR)
	@echo "[TRAIN] Starting training for N=$(TRAIN_N), R=$(R)"
	@if [ "$(USE_LKH)" = "1" ]; then LKH_FLAG="--use_lkh --lkh_exe $(LKH_EXE)"; else LKH_FLAG=""; fi; \
	$(PYWARN) $(PYTHON) -u $(TRAIN_ENTRY) \
		--train_pt $(TRAIN_DATA) \
		--val_pt $(VAL_DATA) \
		--r $(R) \
		--device $(DEVICE) \
		--seed $(SEED) \
		--batch_size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--w_bc $(W_BC) \
		--two_opt_passes $(TWO_OPT_PASSES) \
		--state_mode $(STATE_MODE) \
		--matching_max_used $(MATCHING_MAX_USED) \
		--decode_backend $(DECODE_BACKEND) \
		--exact_time_limit $(EXACT_TIME_LIMIT) \
		--exact_length_weight $(EXACT_LENGTH_WEIGHT) \
		--ckpt_dir $(CKPT_DIR) \
		--log_interval $(LOG_INTERVAL) \
		--val_interval $(VAL_INTERVAL) \
		--save_interval $(SAVE_INTERVAL) \
		--num_workers $(NUM_WORKERS) \
		--num_checkpoints $(NUM_CHECKPOINTS) \
		$$LKH_FLAG \
		--ckpt $(CKPT) 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it" | grep --line-buffered -v "^[[:space:]]*$$" > $(LOG_FILE))

# NeuroLKH C extension compilation (Incremental)
NEUROLKH_SWIG_DIR := src_baselines/neurolkh/SRC_swig
NEUROLKH_SO := $(NEUROLKH_SWIG_DIR)/_LKH.so
NEUROLKH_PY := $(NEUROLKH_SWIG_DIR)/LKH.py
NEUROLKH_SRC := $(filter-out %_wrap.c, $(wildcard $(NEUROLKH_SWIG_DIR)/*.c)) \
                $(wildcard $(NEUROLKH_SWIG_DIR)/*.h) \
                $(NEUROLKH_SWIG_DIR)/LKH.i

$(NEUROLKH_SO) $(NEUROLKH_PY): $(NEUROLKH_SRC)
	@echo "[COMPILE] Checking for swig..."
	@command -v swig >/dev/null 2>&1 || { echo >&2 "[ERROR] swig is not installed. Please install it (e.g., sudo apt-get install swig)."; exit 1; }
	@echo "[COMPILE] Compiling NeuroLKH C extensions (Sources changed)..."
	cd $(NEUROLKH_SWIG_DIR) && \
	swig -python -IINCLUDE LKH.i && \
	$(PYTHON) setup.py build_ext --inplace

compile_neurolkh: $(NEUROLKH_SO)

# Baseline training: NeuroLKH
train_neuraLKH: $(TRAIN_DATA) compile_neurolkh
	@mkdir -p checkpoints/neurolkh_baselines
	@echo "[TRAIN NeuroLKH] Starting baseline training for N=$(TRAIN_N)"
	export PYTHONPATH=$(PWD):$(PWD)/src_baselines:$$PYTHONPATH; \
	$(PYWARN) $(PYTHON) -u src_baselines/train_neuraLKH.py \
		--train_pt $(TRAIN_DATA) \
		--val_pt $(VAL_DATA) \
		--n_epoch $(EPOCHS) \
		--batch_size 20 \
		--num_checkpoints $(NUM_CHECKPOINTS) \
		--save_dir checkpoints/neurolkh_baselines 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it" | grep --line-buffered -v "^[[:space:]]*$$" > $(NEUROLKH_LOG))

# Baseline training: POMO
train_POMO: $(TRAIN_DATA)
	@mkdir -p checkpoints/pomo_baselines
	@echo "[TRAIN POMO] Starting baseline training for N=$(TRAIN_N)"
	export PYTHONPATH=$(PWD):$(PWD)/src_baselines:$$PYTHONPATH; \
	$(PYWARN) $(PYTHON) -u src_baselines/train_POMO.py \
		--train_pt $(TRAIN_DATA) \
		--val_pt $(VAL_DATA) \
		--n_epoch $(EPOCHS) \
		--batch_size 64 \
		--num_checkpoints $(NUM_CHECKPOINTS) \
		--save_dir checkpoints/pomo_baselines 2>&1 | tee >(tr '\r' '\n' | grep --line-buffered -v -E "it/s|s/it" | grep --line-buffered -v "^[[:space:]]*$$" > $(POMO_LOG))

# labels: only run pre-computation for a dataset without starting training
labels: $(TRAIN_DATA) $(VAL_DATA)
	@echo "[LABELS] Pre-computing labels for N=$(TRAIN_N), passes=$(TWO_OPT_PASSES)"
	@if [ "$(USE_LKH)" = "1" ]; then LKH_FLAG="--use_lkh --lkh_exe $(LKH_EXE)"; else LKH_FLAG=""; fi; \
	$(PYWARN) $(PYTHON) $(TRAIN_ENTRY) \
		--train_pt $(TRAIN_DATA) \
		--val_pt $(VAL_DATA) \
		--two_opt_passes $(TWO_OPT_PASSES) \
		--state_mode $(STATE_MODE) \
		--matching_max_used $(MATCHING_MAX_USED) \
		--num_workers $(NUM_WORKERS) \
		$$LKH_FLAG \
		--prepare_labels_only

# Catch-all to generate data for a specific N if not in SIZES
$(DATA_DIR)/N%/train_r_light_pyramid.pt:
	@echo "[AUTO] Missing data for N=$(shell echo $*). Generating now..."
	$(MAKE) all SIZES="$(shell echo $*)"

# Standard academic datasets (Attention Model)
data_std:
	@echo "[STD] Generating standard .pkl using attention-model script for sizes: $(SIZES)"
	@mkdir -p $(DATA_STD_DIR)_pkl
	$(PYWARN) $(PYTHON) $(THIRD_PARTY_DIR)/attention-learn-to-route/generate_data.py \
		--name std --problem tsp --graph_sizes $(SIZES) --dataset_size 1000 \
		--data_dir $(DATA_STD_DIR)_pkl --seed 1234 -f
	@for N in $(SIZES); do \
		echo "[STD] Processing N$$N..."; \
		mkdir -p $(DATA_STD_DIR)/N$${N}; \
		$(PYWARN) $(PYTHON) src/utils/convert_pkl_to_pt.py \
			--input $(DATA_STD_DIR)_pkl/tsp/tsp$${N}_std_seed1234.pkl \
			--output $(DATA_STD_DIR)/N$${N}/test.pt; \
		$(PYWARN) $(PYTHON) src/graph/spanner.py --input $(DATA_STD_DIR)/N$${N}/test.pt --output $(DATA_STD_DIR)/N$${N}/test_spanner.pt --num_workers $(NUM_WORKERS); \
		$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py --input $(DATA_STD_DIR)/N$${N}/test_spanner.pt --output $(DATA_STD_DIR)/N$${N}/test_raw_pyramid.pt --max_points $(MAX_POINTS_PER_LEAF) --max_depth $(MAX_DEPTH) --num_workers $(NUM_WORKERS); \
		$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $(DATA_STD_DIR)/N$${N}/test_raw_pyramid.pt --output $(DATA_STD_DIR)/N$${N}/test_r_light_pyramid.pt --r $(R) --num_workers $(NUM_WORKERS); \
	done

# ==========================================
# 5. Evaluation & Visualization
# ==========================================
STEPS_PER_EPOCH := $(shell echo $$(( ($(TRAIN_SAMPLES) + $(BATCH_SIZE) - 1) / $(BATCH_SIZE) )))
FINAL_STEP      := $(shell echo $$(( $(STEPS_PER_EPOCH) * $(EPOCHS) )))
# Default for eval, vis_pred, eval_lkh
CKPT            ?= $(CKPT_DIR)/ckpt_best.pt

eval:
	@if [ ! -f $(CKPT) ]; then echo "Error: Checkpoint $(CKPT) not found."; exit 1; fi
	@if [ "$(USE_LKH)" = "1" ]; then LKH_FLAG="--use_lkh --lkh_exe $(LKH_EXE)"; else LKH_FLAG=""; fi; \
	$(PYWARN) $(PYTHON) $(TRAIN_ENTRY) \
		--train_pt $(TRAIN_DATA) \
		--val_pt $(VAL_DATA) \
		--r $(R) \
		--device $(DEVICE) \
		--ckpt $(CKPT) \
		--two_opt_passes $(TWO_OPT_PASSES) \
		--decode_backend $(DECODE_BACKEND) \
		--exact_time_limit $(EXACT_TIME_LIMIT) \
		--exact_length_weight $(EXACT_LENGTH_WEIGHT) \
		--num_workers $(NUM_WORKERS) \
		$$LKH_FLAG \
		--eval_only

vis_pred:
	@if [ ! -f $(CKPT) ]; then echo "Error: CheckPT $(CKPT) not found."; exit 1; fi
	@if [ "$(USE_LKH)" = "1" ]; then LKH_FLAG="--use_lkh --lkh_exe $(LKH_EXE)"; else LKH_FLAG=""; fi; \
	$(PYWARN) $(PYTHON) $(VIS_ENTRY) \
		--ckpt $(CKPT) \
		--data_pt $(VAL_DATA) \
		--sample_idx 0 \
		--r $(R) \
		--device $(DEVICE) \
		--two_opt_passes $(TWO_OPT_PASSES) \
		--exact_time_limit $(EXACT_TIME_LIMIT) \
		--exact_length_weight $(EXACT_LENGTH_WEIGHT) \
		--output_dir $(OUT_DIR)/eval \
		--num_workers $(NUM_WORKERS) \
		$(EXACT_FLAG) \
		$$LKH_FLAG

# Deep comparison: Greedy (B1) vs Neural-Guided LKH (B2) vs Pure LKH (B3)
# This generates detailed logs and comparison plots in outputs/eval_lkh/
# Use VIS=1 to enable visualization, default is 0 (faster for batch)
SAMPLE_IDX     ?= 0
SAMPLE_IDX_END ?= 10
VIS            ?= 0

ifeq ($(VIS), 1)
    VIS_FLAG :=
else
    VIS_FLAG := --no_vis
endif

ifeq ($(RUN_EXACT), 1)
    EXACT_FLAG := --run_exact
else
    EXACT_FLAG :=
endif

ifeq ($(strip $(SETTINGS)),)
    SETTINGS_FLAG :=
else
    SETTINGS_FLAG := --settings "$(SETTINGS)"
endif

ifeq ($(strip $(TSPLIB_RUN_TAG)),)
    TSPLIB_RUN_TAG_FLAG :=
else
    TSPLIB_RUN_TAG_FLAG := --run_tag "$(TSPLIB_RUN_TAG)"
endif

ifeq ($(strip $(TSPLIB_SET)),)
    TSPLIB_SET_FLAG :=
else
    TSPLIB_SET_FLAG := --tsplib_set "$(TSPLIB_SET)"
endif

ifeq ($(strip $(TSPLIB_INSTANCES)),)
    TSPLIB_INSTANCES_FLAG :=
else
    TSPLIB_INSTANCES_FLAG := --tsplib_instances "$(TSPLIB_INSTANCES)"
endif

ifeq ($(strip $(DATA_PT)),)
    DATA_PT_FLAG :=
else
    DATA_PT_FLAG := --data_pt "$(DATA_PT)"
endif

ifeq ($(strip $(EVAL_OUTPUT_DIR)),)
    EVAL_OUTPUT_DIR_FLAG :=
else
    EVAL_OUTPUT_DIR_FLAG := --output_dir "$(EVAL_OUTPUT_DIR)"
endif

ifeq ($(strip $(RUN_TAG)),)
    RUN_TAG_FLAG :=
else
    RUN_TAG_FLAG := --run_tag "$(RUN_TAG)"
endif

bench_eval:
	@if [ ! -f $(CKPT) ]; then echo "Error: Checkpoint $(CKPT) not found."; exit 1; fi
	@if [ "$(BENCHMARK)" = "synthetic" ]; then \
		if [ -n "$(DATA_PT)" ]; then \
			DATA_FILE="$(DATA_PT)"; \
		else \
			DATA_FILE="$(DATA_STD_DIR)/N$(SYNTHETIC_N)/test_r_light_pyramid.pt"; \
			if [ ! -f "$$DATA_FILE" ]; then \
				echo "[STD] Standard data not found at $$DATA_FILE. Generating now (N=$(SYNTHETIC_N))..."; \
				$(MAKE) data_std SIZES="$(SYNTHETIC_N)"; \
			fi; \
		fi; \
	fi
	@echo "[BENCH] Evaluating $(BENCHMARK)..."
	@if [ "$(USE_LKH)" = "1" ]; then LKH_FLAG="--use_lkh"; else LKH_FLAG=""; fi; \
	$(PYWARN) $(PYTHON) $(BENCH_EVAL_ENTRY) \
		--benchmark $(BENCHMARK) \
		--ckpt $(CKPT) \
		--r $(R) \
		--device $(DEVICE) \
		--lkh_exe $(LKH_EXE) \
		--num_workers $(NUM_WORKERS) \
		--two_opt_passes $(TWO_OPT_PASSES) \
		--exact_time_limit $(EXACT_TIME_LIMIT) \
		--exact_length_weight $(EXACT_LENGTH_WEIGHT) \
		--synthetic_n $(SYNTHETIC_N) \
		--sample_idx $(SAMPLE_IDX) \
		--sample_idx_end $(SAMPLE_IDX_END) \
		--tsplib_dir $(TSPLIB_DIR) \
		--num_instances $(NUM_INSTANCES) \
		--pomo_ckpt "$(POMO_CKPT)" \
		--neurolkh_ckpt "$(NEUROLKH_CKPT)" \
		$(SETTINGS_FLAG) \
		$(EXACT_FLAG) \
		$(VIS_FLAG) \
		$(DATA_PT_FLAG) \
		$(TSPLIB_SET_FLAG) \
		$(TSPLIB_INSTANCES_FLAG) \
		$(EVAL_OUTPUT_DIR_FLAG) \
		$(RUN_TAG_FLAG) \
		$$LKH_FLAG

eval_lkh:
	@if [ ! -f $(CKPT) ]; then echo "Error: CheckPT $(CKPT) not found."; exit 1; fi
	STD_DATA=$(DATA_STD_DIR)/N$(EVAL_LKH_N)/test_r_light_pyramid.pt; \
	if [ ! -f $$STD_DATA ]; then \
		echo "[STD] Standard data not found at $$STD_DATA. Generating now (N=$(EVAL_LKH_N))..."; \
		$(MAKE) data_std SIZES="$(EVAL_LKH_N)"; \
	fi; \
	$(MAKE) bench_eval \
		BENCHMARK=synthetic \
		SYNTHETIC_N=$(EVAL_LKH_N) \
		DATA_PT="$$STD_DATA" \
		EVAL_OUTPUT_DIR=$(OUT_DIR)/eval_lkh \
		RUN_TAG= \
		TSPLIB_SET= \
		TSPLIB_INSTANCES=

# TSPLIB Evaluation: Using hardcoded LKH results from the paper for large instances.
# NUM_INSTANCES: report top K largest instances
NUM_INSTANCES  ?= 10
POMO_CKPT      ?= 
NEUROLKH_CKPT  ?= 
eval_tsplib:
	@if [ ! -f $(CKPT) ]; then echo "Error: Checkpoint $(CKPT) not found."; exit 1; fi
	@echo "[TSPLIB] Evaluating TSPLIB benchmark..."
	$(MAKE) bench_eval \
		BENCHMARK=tsplib \
		SYNTHETIC_N=$(SYNTHETIC_N) \
		EVAL_OUTPUT_DIR=$(TSPLIB_SAVE_DIR) \
		RUN_TAG=$(TSPLIB_RUN_TAG) \
		TSPLIB_SET="$(TSPLIB_SET)" \
		TSPLIB_INSTANCES="$(TSPLIB_INSTANCES)"

# ==========================================
# 6. Utilities
# ==========================================
clean:
	rm -rf $(OUT_DIR)

test_bottomup: $(DATA_DIR)/N$(TEST_N)/$(TEST_SPLIT)_r_light_pyramid.pt
	$(PYWARN) $(PYTHON) tests/test_bottom_up_process.py \
		--data_pt $< \
		--idx 0 \
		--r $(R) \
		--device $(DEVICE)

# ---- Bottom-up debug run ----
TEST_N ?= 50
TEST_SPLIT ?= train
TEST_IDX ?= 0
# DEVICE ?= cuda (already defined globally)
