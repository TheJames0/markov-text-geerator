CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Isrc -MMD
LDFLAGS :=
TARGET := textgen
UDPIPE_BIN := udpipe
UDPIPE_SRC := /tmp/udpipe
MODEL_DIR := models
MODEL_FILE := $(MODEL_DIR)/english.udpipe

HAS_FS := $(shell echo "int main(){}" | $(CXX) -std=c++17 -x c++ -lstdc++fs - -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(HAS_FS), 1)
    LDFLAGS += -lstdc++fs
endif

DIM ?= 50
EMBEDDINGS_ZIP_URL := https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
EMBEDDINGS_ZIP := glove.6B.zip
EMBEDDINGS_FILE := glove.6B.$(DIM)d.txt

UDPIPE_REPO := https://github.com/ufal/udpipe.git
MODEL_URL := https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/8d51e3d3-1f87-47fe-8975-c197ddf7907b/content

.PHONY: all clean clean-all run embeddings build-udpipe model

all: $(TARGET)

$(TARGET): src/main.cpp $(wildcard src/*.hpp)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

-include $(wildcard src/*.d)

embeddings: $(EMBEDDINGS_FILE)

$(EMBEDDINGS_FILE):
	@echo "Downloading GloVe 6B embeddings (~862MB zip, extracting $(DIM)d file)..."
	@curl -L --connect-timeout 30 --max-time 3600 "$(EMBEDDINGS_ZIP_URL)" -o "$(EMBEDDINGS_ZIP)" --progress-bar
	@unzip -o "$(EMBEDDINGS_ZIP)" "$(EMBEDDINGS_FILE)"
	@rm -f "$(EMBEDDINGS_ZIP)"
	@echo "Downloaded: $(EMBEDDINGS_FILE)"

build-udpipe: $(UDPIPE_BIN)

$(UDPIPE_BIN):
	@echo "Building UDPipe from $(UDPIPE_REPO)..."
	@if [ ! -d "$(UDPIPE_SRC)" ]; then \
		git clone --depth 1 $(UDPIPE_REPO) $(UDPIPE_SRC); \
	fi
	$(MAKE) -C $(UDPIPE_SRC)/src
	cp $(UDPIPE_SRC)/src/udpipe $(UDPIPE_BIN)
	@echo "Built: $(UDPIPE_BIN)"

model: $(MODEL_FILE)

$(MODEL_FILE):
	@echo "Downloading UDPipe English model..."
	@mkdir -p $(MODEL_DIR)
	@curl -L --connect-timeout 30 --max-time 300 \
		"$(MODEL_URL)" -o "$(MODEL_FILE)" --progress-bar
	@echo "Downloaded: $(MODEL_FILE)"

clean:
	rm -f $(TARGET)

clean-all: clean
	rm -f glove.6B.*.txt $(UDPIPE_BIN)
	rm -rf $(MODEL_DIR)

run: $(TARGET) $(UDPIPE_BIN) $(MODEL_FILE)
	./$(TARGET) data glove.6B.$(DIM)d.txt
