DEMOFILES=demo00.cpp demo01.cpp demo02.cpp demo03.cpp

CXX=g++ -m64
CXXFLAGS=-fopenmp

#architecture specific flags
ARCHFLAGS=-march=corei7-avx -mavx
OPTFLAGS= -O3  -ftree-vectorize 
ESVMFLAGS=-D ESVM_PERFORMANCE_COUNTERS -D ESVM_CONFIG_ALIGN_MEMORY
DEBUGFLAGS=-g -rdynamic
#-g -rdynamic help in printing a good stack trace
ISPC=../common/ispc-64bit-linux
# architecture specific flags
ISPCARCHFLAGS=--target=avx --arch=x86-64 --cpu=corei7-avx 
ISPCFLAGS=-O2 --math-lib=fast --opt=fast-masked-vload --wno-perf

APP_NAME=$(DEMOFILES:.cpp=.out)
OBJDIR=objs
ISPC_COMMONDIR=../common
ESVM_DIR=../internal
ESVM_CXX=$(ESVM_DIR)/esvm.cpp $(ESVM_DIR)/esvm_utils.cpp $(ESVM_DIR)/resize.cpp
ESVM_ISPC=$(ESVM_DIR)/hog.ispc $(ESVM_DIR)/conv3D.ispc

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

CXXINCLUDE=-I$(ISPC_COMMONDIR) -I$(OBJDIR) -I$(ESVM_DIR) `pkg-config --cflags opencv`
LDFLAGS=-lm `pkg-config --libs opencv` $(TASKSYS_LIB)

default: all

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean-prof:
		/bin/rm -f $(OBJDIR)/*.o  $(OBJDIR)/*.h *~ $(APP_NAME)

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

ESVM_CXX_OBJ=$(addprefix $(OBJDIR)/, $(subst $(ESVM_DIR)/,, $(ESVM_CXX:.cpp=.o)))
ESVM_ISPC_OBJ=$(addprefix $(OBJDIR)/, $(subst $(ESVM_DIR)/,, $(ESVM_ISPC:.ispc=_ispc.o)))
ESVM_ISPC_HEADER=$(addprefix $(OBJDIR)/, $(subst $(ESVM_DIR)/,, $(ESVM_ISPC:.ispc=_ispc.h)))

DEMO_OBJ=$(addprefix $(OBJDIR)/, $(subst $(ESVM_DIR)/,,$(DEMOFILES:.cpp=.o)))
OBJS=$(ESVM_ISPC_OBJ) $(ESVM_CXX_OBJ) $(TASKSYS_OBJ)

all: dirs $(APP_NAME) 

$(APP_NAME): $(OBJS) $(DEMO_OBJ)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(addprefix $(OBJDIR)/,$(subst .out,.o, $@)) $(CXXINCLUDE) $(LDFLAGS) $(ARCHFLAGS) $(OPTFLAGS) $(ESVMFLAGS) $(DEBUGFLAGS)

$(OBJDIR)/%_ispc.o $(OBJDIR)/%_ispc.h: $(ESVM_DIR)/%.ispc
	$(ISPC) $(ISPCFLAGS) $(ISPCARCHFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

$(OBJDIR)/%.o: $(ESVM_DIR)/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@ $(CXXINCLUDE) $(LDFLAGS) $(ARCHFLAGS) $(OPTFLAGS) $(ESVMFLAGS) $(DEBUGFLAGS)

$(OBJDIR)/%.o: $(ISPC_COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(CXXINCLUDE) $(LDFLAGS) $(ARCHFLAGS) $(OPTFLAGS) $(ESVMFLAGS) $(DEBUGFLAGS)
	
$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(CXXINCLUDE) $(LDFLAGS) $(ARCHFLAGS) $(OPTFLAGS) $(ESVMFLAGS) $(DEBUGFLAGS)
