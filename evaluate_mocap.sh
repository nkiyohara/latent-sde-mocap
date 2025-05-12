uv run evaluate_mocap.py \
  --run-names atomic-glade-71 youthful-water-67 glowing-cloud-63 \
              dainty-rain-63 snowy-glade-63 \
  --step 5000 \
  --dts 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0\
  --device cpu
