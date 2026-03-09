import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Load Edge Impulse Model
# -------------------------------------------------------

# load model
interpreter = tf.lite.Interpreter(
    model_path="output/ei-seizuredetection3-classifier-tensorflow-lite-int8-quantized-model.3.lite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

input_scale, input_zero_point = input_details[0]['quantization']
out_scale, out_zero_point = output_details[0]['quantization']

# load flattened feature set from Edge Impulse
X = np.load("output/ei-seizuredetection3-flatten-X_testing.2.npy")
print(X.shape)

# -------------------------------------------------------
# Run Inference
# -------------------------------------------------------
probabilities = []

for row in X:
    # Quantize the input
    quantized = (row / input_scale + input_zero_point).astype(np.int8)
    sample = np.expand_dims(quantized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # convert output to int
    logit0 = int(output[0][0])
    logit1 = int(output[0][1])

    # Dequantize the output
    logit0 = out_scale * (logit0 - out_zero_point)
    logit1 = out_scale * (logit1 - out_zero_point)

    #softmax
    exp0 = np.exp(logit0)
    exp1 = np.exp(logit1)

    prob_seizure = exp1 / (exp0 + exp1)

    probabilities.append(prob_seizure)

probabilities = np.array(probabilities)
print("First 50 probabilities:")
print(probabilities[:50])

y  = np.load("output/ei-seizuredetection3-flatten-y_testing.2.npy")
if len(y.shape) > 1:
    y = np.argmax(y, axis=1)
print("Mean seizure probability: ", probabilities[y==1].mean())
print("Mean normal probability: ", probabilities[y==0].mean())

plt.plot(probabilities)
plt.title("Seizure probability over time")
plt.show()

# -------------------------------------------------------
# Prediction-Lite Processing
# -------------------------------------------------------

df = pd.DataFrame()
df["probability"] = probabilities

# Moving average smoothing
window = 30
df["smooth"] = df["probability"].rolling(window).mean()
# plt.plot(df["smooth"][1000:2000])
# plt.show()

# Abnormality slope
df["slope"] = np.gradient(df["smooth"])

#Early warning rule
df["warning"] = (df["smooth"] > 0.48) & (df["slope"] > 0.001)

# Save results
df.to_csv("seizure_prediction_results.csv", index=False)
print("Saved results to seizure_prediction_results.csv")

# Plot results
plt.figure(figsize=(12,5))

plt.plot(df["probability"], alpha=0.3, label="Raw probability")
plt.plot(df["smooth"], linewidth=2, label="Smoothed probaility")

plt.title("Seizure probability over time")
plt.xlabel("Time step")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Plot slope
plt.figure(figsize=(12,5))
plt.plot(df["slope"], color="purple")
plt.title("Abnormality slope")
plt.xlabel("Time step")
plt.ylabel("Slope")
plt.show()

# Plot warnings
plt.figure(figsize=(12,5))
plt.plot(df["smooth"], label="Smoothed probability")
warning_indices = np.where(df["warning"] == True)[0]
plt.scatter(warning_indices,
            df["smooth"].iloc[warning_indices],
            color="red",
            label="Early warning")

plt.legend()
plt.title("Early seizure prediction signals")
plt.show()

print("Total early warnings detected: ", df["warning"].sum())