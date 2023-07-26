
import sys
sys.path.append('../')
import json
import os
import gi
import pyds
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.bus_call import bus_call
import time
import cv2
import numpy as np


PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
pgie_classes_str = ["Person"]
output = {}

start_time = time.time()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', default='/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.h264')
    parser.add_argument('-m', '--model', help='path to .engine file')
    parser.add_argument('-c', '--conf', help='path to txt file')
    args = parser.parse_args()
    return args

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    print('obj', obj_name)
    print('top', top)
    print('left', left)
    print("width", width)
    print('height', height)
    output[obj_name] = [(top, left), (width, height)]
    

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    print()
    print("batch_meta", dir(batch_meta))
    print('l_frame',  dir(l_frame), '\n')
    print('l_frame.data',  dir(l_frame.data), '\n')
    print()


    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            print('frame_meta', dir(frame_meta))
        except StopIteration:
            print("Itreation stopped")
            exit()
            break

        #Intiallizing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_BICYCLE:0,
            PGIE_CLASS_ID_ROADSIGN:0
        }
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
            # convert python array into numpy array format in the copy mode.
            frame_copy = np.array(n_frame, copy=True, order='C')
            # convert the array into cv2 default color format
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite('img.jpg', frame_copy)
            # if is_aarch64(): # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                # pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                    #  The original array cannot be accessed after this call.

            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8) #0.8 is alpha (opacity)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	


def main(args):
    Gst.init(None)
    pipeline = Gst.Pipeline()
    
    #define elements
    source = Gst.ElementFactory.make("filesrc", "file-source")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")

    #set property of elements
    source.set_property('location', args.video)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batched-push-timeout', 4000000)
    streammux.set_property('batch-size', 1)
    pgie.set_property('config-file-path', "dstest1_pgie_config.txt")
    pgie.set_property('model-engine-file', args.model)
    sink.set_property('sync', 0)

    #.Add elements to the pipeline
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    #link the elements together : file-source -> h264-parser -> nvh264-decoder -> nvinfer -> nvvidconv -> nvosd -> video-renderer
    srcpad = decoder.get_static_pad("src")
    sinkpad = streammux.get_request_pad("sink_0")
    
    source.link(h264parser)
    h264parser.link(decoder)
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to the sink pad of the osd element, since by that time, the buffer would have had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))