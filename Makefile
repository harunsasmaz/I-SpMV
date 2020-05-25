CXX      := g++
CXXFLAGS := -std=c++14
CXXFLAGS += 
LDFLAGS  +=  
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)
TARGET   := spmv
INCLUDE  := -Iinclude/
SRC      :=                      \
   $(wildcard src/*.cpp)         \


OBJECTS := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
OS := $(shell uname)
TARGET_SIZE=64
ALL_LDFLAGS = 
LIBRARIES += 

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(APP_DIR)/$(TARGET): $(OBJECTS) $(OBJECTS_CUDA)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(ALL_LDFLAGS) $(INCLUDE) $(OBJECTS) $(LDFLAGS) -o $(APP_DIR)/$(TARGET)

.PHONY: all build clean debug release

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O3
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*
