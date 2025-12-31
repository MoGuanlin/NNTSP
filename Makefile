# Makefile (place at repo root)
PYWARN ?= PYTHONWARNINGS="ignore::FutureWarning"

PYTHON ?= python

DATA_DIR ?= data
OUT_DIR  ?= outputs

# Problem sizes (space-separated)
SIZES ?= 20 50

# Data generation
GRID_SIZE ?= 10000
TRAIN_SAMPLES ?= 200
SEED ?= 1234
ENSURE_UNIQUE ?= 1  # 1: enforce unique points per instance; 0: allow duplicates

# Quadtree build
MAX_POINTS_PER_LEAF ?= 4
MAX_DEPTH ?= 20

# r-light pruning parameter
R ?= 4

# Visualization
SAMPLE_IDX ?= 0
MAX_DEPTH_DRAW ?= 20

# Helpers for "20 50" -> "20,50"
empty :=
space := $(empty) $(empty)
comma := ,
SIZES_CSV := $(subst $(space),$(comma),$(strip $(SIZES)))

.PHONY: all deps data spanner raw rlight vis clean

all: data spanner raw rlight vis

# Optional: install python deps (run manually if needed)
deps:
	$(PYWARN) $(PYTHON) -m pip install --upgrade pip
	$(PYWARN) $(PYTHON) -m pip install scipy scikit-learn matplotlib torch torch-geometric tqdm

# Generate points datasets: data/N*/train.pt, val.pt, test.pt
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

# Build spanner graphs for train/val/test
spanner: $(foreach N,$(SIZES),\
	$(DATA_DIR)/N$(N)/train_spanner.pt \
	$(DATA_DIR)/N$(N)/val_spanner.pt \
	$(DATA_DIR)/N$(N)/test_spanner.pt)

$(DATA_DIR)/N%/train_spanner.pt: $(DATA_DIR)/N%/train.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@

$(DATA_DIR)/N%/val_spanner.pt: $(DATA_DIR)/N%/val.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@

$(DATA_DIR)/N%/test_spanner.pt: $(DATA_DIR)/N%/test.pt
	@echo "[SPANNER] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/spanner.py --input $< --output $@

# Build raw quadtree pyramid (List[Data]) for train/val/test
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
		--max_depth $(MAX_DEPTH)

$(DATA_DIR)/N%/val_raw_pyramid.pt: $(DATA_DIR)/N%/val_spanner.pt
	@echo "[RAW] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py \
		--input $< \
		--output $@ \
		--max_points $(MAX_POINTS_PER_LEAF) \
		--max_depth $(MAX_DEPTH)

$(DATA_DIR)/N%/test_raw_pyramid.pt: $(DATA_DIR)/N%/test_spanner.pt
	@echo "[RAW] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/build_raw_pyramid.py \
		--input $< \
		--output $@ \
		--max_points $(MAX_POINTS_PER_LEAF) \
		--max_depth $(MAX_DEPTH)

# Prune to r-light pyramid for train/val/test
rlight: $(foreach N,$(SIZES),\
	$(DATA_DIR)/N$(N)/train_r_light_pyramid.pt \
	$(DATA_DIR)/N$(N)/val_r_light_pyramid.pt \
	$(DATA_DIR)/N$(N)/test_r_light_pyramid.pt)

$(DATA_DIR)/N%/train_r_light_pyramid.pt: $(DATA_DIR)/N%/train_raw_pyramid.pt
	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R)

$(DATA_DIR)/N%/val_r_light_pyramid.pt: $(DATA_DIR)/N%/val_raw_pyramid.pt
	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R)

$(DATA_DIR)/N%/test_r_light_pyramid.pt: $(DATA_DIR)/N%/test_raw_pyramid.pt
	@echo "[RLIGHT r=$(R)] $< -> $@"
	$(PYWARN) $(PYTHON) src/graph/prune_pyramid.py --input $< --output $@ --r $(R)

# Visualize one sample (default: train split) for each N
vis: $(foreach N,$(SIZES),$(OUT_DIR)/vis_N$(N)_train.png)

$(OUT_DIR)/vis_N%_train.png: $(DATA_DIR)/N%/train_r_light_pyramid.pt
	@mkdir -p $(OUT_DIR)
	@echo "[VIS] $< -> $@"
	$(PYWARN) $(PYTHON) src/visualization/visualize_pyramid.py \
		--input $< \
		--sample_idx $(SAMPLE_IDX) \
		--output $@ \
		--max_depth_draw $(MAX_DEPTH_DRAW)

clean:
	rm -rf $(OUT_DIR)

# ---- Bottom-up debug run ----
TEST_N ?= 50
TEST_SPLIT ?= train
TEST_IDX ?= 0
DEVICE ?= cuda

.PHONY: test_bottomup
test_bottomup:
	$(PYWARN) $(PYTHON) src/test/test_bottom_up_process.py \
		--data_pt $(DATA_DIR)/N$(TEST_N)/$(TEST_SPLIT)_r_light_pyramid.pt \
		--idx $(TEST_IDX) \
		--r $(R) \
		--device $(DEVICE)

