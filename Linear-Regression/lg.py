import numpy as np

attrs = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL','RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def read_train_csv(fileName, N):

    data = np.genfromtxt(fileName, delimiter=',', skip_header=1)[:, 3:].astype(float)
    data = data.reshape(12, -1, 18, 24) # 12 months, 20 days per month, 18 features per day, 24 hours per day
    train_X, train_Y = get_N_hours_feat(data[0], N)

    for i in range(1, 12):
        X, Y = get_N_hours_feat(data[i], N)
        train_X = np.concatenate((train_X, X), axis=0)
        train_Y = np.concatenate((train_Y, Y), axis=0)

    return train_X, train_Y

def read_test_csv(fileName, N):

    test_days = DAYS-20
    cumul_days = [sum(test_days[:i]) for i in range(1, 13)]
    data = np.genfromtxt(fileName, delimiter=',', skip_header=1)[:, 3:].astype(float).reshape(-1, 18, 24)

    test_X, test_Y = get_N_hours_feat(data[:cumul_days[0]], N)
    
    for i in range(1, 12):
        X, Y = get_N_hours_feat(data[cumul_days[i-1]:cumul_days[i]], N)
        test_X = np.concatenate((test_X, X), axis=0)
        test_Y = np.concatenate((test_Y, Y), axis=0)
    
    return test_X, test_Y

def get_N_hours_feat(month_data, N):
    # month_data.shape = (num_of_date, 18, 24)

    data = month_data.transpose((0, 2, 1)).reshape(-1, 18)
    label = month_data.transpose((1, 0, 2)).reshape(18, -1)[9]
    total_hours = len(label)
    
    feats = np.array([])
    for i in range(total_hours-N):
        cur_feat = np.append(data[i:i+N].flatten(), [1])   # add w0, to discuss without w0, please comment this line! 
        # ???????????????????????????????????????????????????????????
        feats = np.concatenate([feats, cur_feat], axis=0)
        #cur_feat = data[i:i+N].flatten()
        #feats = np.concatenate([feats, cur_feat], axis=0)


    label = label[N:]

    feats = feats.reshape(-1, N*18+1)  # to discuss without w0, please change to N*18!
    #feats = feats.reshape(-1, N*18)
    
    return feats, label

class Linear_Regression(object):
    
    def __init__(self):
        pass

    def train(self, X, Y):
        #TODO: the shape of W should be number of features
        C = X
        CT = np.transpose(C)
        W = np.matmul(np.matmul(np.linalg.inv(np.matmul(CT,C)),CT),Y)
        # W = (X.T * X)inv * X.T * Y
        self.W = W
        
    def predict(self, X):
        pred_X = np.matmul(X,self.W)
        return pred_X

def MSE(pred_Y, real_Y):
    #TODO: mean square error 
    n = len(pred_Y)
    sum = 0
    for i in range(n):
        sum += pow(pred_Y[i] - real_Y[i],2)
    error = sum/n
    return error

