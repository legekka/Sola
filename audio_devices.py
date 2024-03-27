import sounddevice as sd                

devices = sd.query_devices()

print(devices)

for (i, device) in enumerate(devices):
    if "Line 1 (Virtual Audio Cable)" in device["name"] and device["hostapi"] == 0 and device["max_output_channels"] == 2:
        print(f"Device {i}: {device['name']}")
        output_id = i
        break

