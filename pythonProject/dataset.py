import os
import cv2
import time

num = int(input("The number of new labels you want to add: "))
number_images = 7

# Create the "Dataset" directory if it doesn't exist
os.makedirs("Dataset", exist_ok=True)

for i in range(num):
    name = input("Name of the label: ")
    labelname = os.path.join("Dataset", name)

    # Create the label directory if it doesn't exist
    os.makedirs(labelname, exist_ok=True)

    # Open the labels.txt file in append mode to avoid overwriting
    with open("Dataset\\labels.txt", "a") as label:  # Use "a" for appending
        label.write(f"{i} {name}\n")  # Add a newline for better readability

    cap = cv2.VideoCapture(0)
    for imgnum in range(number_images):
        print("Collecting images {}".format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(labelname, f"Num {imgnum+1} {name}.jpg")
        cv2.imwrite(imgname, frame)
        cv2.imshow("frame", frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
