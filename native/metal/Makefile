# Metal bridge build — macOS only
# Requires: Xcode command line tools (xcode-select --install)
#
# Usage:
#   make            — build dylib and metallib
#   make clean      — remove build artifacts
#   make install    — copy to .NET output directory

CC = clang
METAL = xcrun metal
METALLIB = xcrun metallib

CFLAGS = -shared -fPIC -O2 -fobjc-arc \
         -framework Metal \
         -framework MetalPerformanceShaders \
         -framework Foundation

DYLIB = libmetal_bridge.dylib
METALLIB_OUT = kernels.metallib

.PHONY: all clean install

all: $(DYLIB) $(METALLIB_OUT)

$(DYLIB): metal_bridge.m metal_bridge.h
	$(CC) $(CFLAGS) -o $(DYLIB) metal_bridge.m

# Note: the add kernel is compiled from embedded source in metal_bridge.m
# This metallib is for additional custom kernels loaded at runtime.
kernels.air: kernels.metal
	$(METAL) -c kernels.metal -o kernels.air

$(METALLIB_OUT): kernels.air
	$(METALLIB) kernels.air -o $(METALLIB_OUT)

clean:
	rm -f $(DYLIB) $(METALLIB_OUT) kernels.air

# Copy to the .NET project output — adjust path as needed
DOTNET_OUT ?= ../../src/Minerva.Interop.Poc/bin/Debug/net10.0

install: all
	@mkdir -p $(DOTNET_OUT)
	cp $(DYLIB) $(DOTNET_OUT)/
	cp $(METALLIB_OUT) $(DOTNET_OUT)/
	@echo "Installed to $(DOTNET_OUT)"
