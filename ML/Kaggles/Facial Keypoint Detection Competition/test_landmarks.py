import face_alignment
from skimage import io
import matplotlib.pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
input = io.imread('mee.jpg')
preds = fa.get_landmarks(input)
print(preds[0].shape)
plt.imshow(input)
plt.plot(preds[0][:, 0], preds[0][:, 1], "ro")
plt.show()
#plt.savefig('myimage.png')
