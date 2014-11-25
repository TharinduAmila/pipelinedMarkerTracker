LOCAL_PATH := $(call my-dir)
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=off
OPENCV_LIB_TYPE:=STATIC
include $(CLEAR_VARS)
include ../sdk/native/jni/OpenCV.mk
LOCAL_CPPFLAGS := -lpthread -frtti -fexceptions
LOCAL_MODULE    := threaded
LOCAL_SRC_FILES := threaded.cpp
LOCAL_LDLIBS    += -lm -llog -landroid
LOCAL_STATIC_LIBRARIES += android_native_app_glue

include $(BUILD_SHARED_LIBRARY)
$(call import-module,android/native_app_glue)
