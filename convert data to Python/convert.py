import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()
eng.Main_TrainModel(nargout=0)

inputX = np.double(eng.workspace['inputX'])
inputY = np.double(eng.workspace['inputY'])
"""testData = np.double(eng.workspace['testData'])
testLabel = np.double(eng.workspace['testLabel'])
trainData = np.double(eng.workspace['trainData'])
trainLabel = np.double(eng.workspace['trainLabel'])
cntTrainFileName = eng.workspace['cntTrainFileName']
cntTestFileName = eng.workspace['cntTestFileName']"""