# Oyster Weight Classifier KNN version

This algorithm takes in inputs oyster pictures of variety "huîtres Marennes Oléros" and classify them by their weight in 4 classes.
It uses a KNN Classifier to do this, it begin to create an array of pixels with images input and for each cut the background. After that, it looks at every rows first and compared distance of the most distant pixels and keep the maximum (corresponding to the longer of oysters). Then, it does the same with columns (corresponding to the larger of oysters). Finally it appends this dimensions to an array for every oysters with their weights respective, in order to train the model. This array is used in a KNN Classifier which classifying 4 classes. Some tests are implemented in the end to validate the code. 
The code is initially used on google collab with language python.
