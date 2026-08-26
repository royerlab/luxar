#### Gallery captures now report under-filled tiles

The gallery harness logs both final subject span and lit screen area for every
framing path, and warns when either falls below the measured README-gallery
baseline. The diagnostic reads the captured still itself and never fails or
adds another screenshot to a gallery run.
