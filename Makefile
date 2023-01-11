
CC        := gcc
LD 	  := gcc

MODULES   := src/LELib
SRC_DIR   := $(addprefix ./, $(MODULES))
SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       := $(patsubst %.cpp,%.o,$(SRC))
INCLUDES  := $(addprefix -I,$(SRC_DIR))

vpath %.cpp $(SRC_DIR)

OPENCV_LIB_PATH := -L./../opencv/public/libraries/release
DEBUG_TBB_LIB_PATH := -L./lib/tbb/debug
RELEASE_TBB_LIB_PATH := -L./lib/tbb/release
OPENCV_INCLUDE_PATH := -I./lib/opencv/include/
INCLUDEPATH := -I./lib -I./src/LELib -I./lib/tbb/include

DEBUG_BUILD_DIR := ./build/debug
RELEASE_BUILD_DIR := ./build/release
# -m64 -fstack-protector -Wtrampolines
CXXFLAGS = -Ofast -fPIC -Wall -Wno-deprecated -DNOMINMAX -std=gnu++11 -DNDEBUG ${INCLUDEPATH} ${OPENCV_INCLUDE_PATH}
# NDEBUG: standard flag used in code or standard libraries for eliminating unwanted code during compilation.
# stack-protector for debugging the stack memory which gets exposed due to error.
# -m64: for run platform to be 64 bit.
# -Wtrampolines: to trigger warnings if system level flags are accessed.
# -fvisibility=hidden: for hiding visibility of functions outside library, except global ones. 
# [global, _T, _U, _G: nm -g => lists exposed fns, G ones are global.]

DEBUG_LDFLAGS = -shared ${DEBUG_TBB_LIB_PATH} -ltbb_debug ${OPENCV_LIB_PATH} -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_imgcodecs -z relro -z now

RELEASE_LDFLAGS = -shared ${RELEASE_TBB_LIB_PATH} -ltbb ${OPENCV_LIB_PATH} -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_imgcodecs -z relro -z now

define make-goal
$1/%.o: %.cpp
	$(CC) -c $(CXXFLAGS) $(INCLUDES) -c $$< -o $$@
endef

all: debug_build release_build

# Target for Lattice Extractor Shared Object
debug_build: $(OBJ)
	# $(CC) -c -fPIC $(CXXFLAGS) $(INCLUDES) src/testapp/main.cpp 
	$(LD) $(DEBUG_LDFLAGS) $(OBJ) -o libextractor.so
	mkdir -p ${DEBUG_BUILD_DIR}
	mv libextractor.so $(DEBUG_BUILD_DIR)/

release_build: $(OBJ)
	$(LD) $(RELEASE_LDFLAGS) $(OBJ) -o libextractor.so
	mkdir -p ${RELEASE_BUILD_DIR}
	mv libextractor.so $(RELEASE_BUILD_DIR)/

%.o: %.cpp
	$(CC) -c $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# Target for Lattice Extractor Exe
latticeExtracterExe: main.o $(OBJ)  
	$(LD) ${RELEASE_LDFLAGS} -o $@ $^ ${RELEASE_LDFLAGS}
	mkdir -p ${RELEASE_BUILD_DIR}
	mv latticeExtracter $(RELEASE_BUILD_DIR)/

main.o: src/testapp/main.cpp
	$(CC) -c $(CXXFLAGS) $<

# Testing constructing Lattice Extractor
latticeExtracter.so: DetectLatticeInImage.o $(OBJ)
	mkdir -p ${RELEASE_BUILD_DIR}
	$(LD) main.o $(OBJ) -o $(RELEASE_BUILD_DIR)/libextractor.so	

DetectLatticeInImage.o: src/testapp/DetectLatticeInImage.cpp
	$(CC) -c $(CXXFLAGS) $<

clean:
	rm -rf $(OBJ); rm main.o;
