In order to use replicate our method, run the detection model on laptop with a GPU. We connected to the Pi using an SSH.

Using the "camera_trans_udp.py" script, get the Raspberry Pi camera stream directly to the laptop running the detection model by 
tweaking the UDP IP and PORT. Let your detection model run on this feed.

Then, in a separate terminal, run "camera_udp_thread.py" to forward the inference message from the model to local UDP which the 
control script will listen to. 

In a third terminal, run the "cutting_losses_new.py" script that will send movement orders to the Arduino based on the offset of
the object detected.

This ensures that the Raspberry Pi's Linux manages the threading instead of tussling with Python's abysmal architecture for that.

Load the "movement.ino" sketch on your Arduino (we used an Arduino Mega), and manage the pin numbers.
