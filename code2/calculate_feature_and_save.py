from imageProcessModules import *

image, label = importAndProcess("train",(224,224))
image = list2image(image)
model = importResnet()

feature_list = []
for i,image_torch in enumerate(image):
    feature_vector = computeFeature(model,image_torch)
    print("Feature Calculated {}/{}".format(i, image.shape[0]))
    feature_list.append(feature_vector)

#output is already normalized in resnet model
feature_normalized = featureNormalize(feature_list)

label_lookup = labelLookUp(label)

label_int = label2int(label,label_lookup)

saveObject(feature_normalized,file_name = "feature_vector.cs484")
saveObject(label_lookup,file_name="label_lookup.cs484")
saveObject(label_int,file_name="label_int.cs484")