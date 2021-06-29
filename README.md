# Motor-Bearing-Monitoring-NN
Electric motor bearing monitoring based on sound utilizing neural networks.

This code demonstrates detection of a faulty bearing in an electric motor by the sound it makes. Detecting faults in industrial environment is important because failure in some critical component might can cause the entire plant to halt. Production losses of downtime can be easily millions of euros. It is possible to detect faults by measuring the vibration of the motors shaft or acceleration. Advantage of sound based detection is cheap cost of microphones. They do not require direct contact with the motor and are therefore more easier to install. However, loud background noise typical in industrial plants can cause problems to sound detection.

Sound of the motors was recorded earlier and sound processing was continued with using MATLAB. Different frequencies of sound was then extracted from the signal using Fast Fourier Transform (FFT). By plotting frequency domain faulty signal can be easily recognized by its 3300 Hz fault frequency and higher amplitude.

Advantages of machine learning:
- Can be trained to detect different kind of fault categories like bearing, stator winding, load and rotor bar
- Can be trained to analyze different signals than sound like vibration for instance
- Method can be applied to many other applications than electric motors 

Video about laboratory setting of normal and faulty motor. Detecting obvious fault by listening is easy. See if you can hear which motor is healty and which one is faulty.
https://www.youtube.com/watch?v=YBFgkpnamAI

Repository contains MATLAB code and pre-recorded 5 second samples of faulty motor and normal motor in MATLAB audio format.

Prerequisites
- Matlab
- Deep learning toolbox
