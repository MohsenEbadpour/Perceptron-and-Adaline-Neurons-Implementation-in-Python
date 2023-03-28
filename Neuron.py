from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
from sklearn.metrics import mean_squared_error


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import pandas as pd 
from datetime import datetime
import os


sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = [9, 6]


def CalculateMetricsAndPlot(true_label, predicted_label,color="Blues",text="",path=""):
    CM = confusion_matrix(true_label, predicted_label)
    acc = round(accuracy_score(true_label,predicted_label)*100,2)
    precision = round(precision_score(true_label,predicted_label, average='macro'),2)
    if text == "":
        sns.heatmap(CM ,annot=True, cmap=color, fmt='g').set_title("Confusion Matrix |Accuracy={0}% |Precision={1}".format(acc,precision))
    else :
        sns.heatmap(CM ,annot=True, cmap=color, fmt='g').set_title("Confusion Matrix |Accuracy={0}% |Precision={1} |{2}".format(acc,precision,text))
    
    if path:
        plt.savefig(path + '/confusion-matrix.png')
        
    plt.show()
    plt.clf()
    
    
def ReLU(x):    
    return x * (x > 0)

def ReLU_Deriv(x):
    return np.array(x > 0).astype(int)
    
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_Deriv(x):
     return np.exp(-x)/((1 + np.exp(-x))**2)
  
def TanH(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def TanH_Deriv(x):
    return 1-TanH(x)**2

def Linear(x):
    return x

def Linear_Deriv(x):
    return 1



class Neuron:
    weights = np.array([]) 
    weights_history = np.array([]) 
    
    loss = np.array([]) 
    val_loss = np.array([]) 
    
    miss_classified_train = np.array([])
    miss_classified_valid = np.array([])
    
    acc = np.array([]) 
    val_acc = np.array([]) 
    
    order = 1 
    neuron_type = "A" 
    
    def __init__(self,neuron_type="P",order=1,activation="T",features=2):
        if not (order in [1,2] and neuron_type in ["A","P"] and activation in ["R","S","T","L"]):
            raise ValueError("Given Parameter is not acceptable: order={0}, neuron type={1}, activation={2}".format(order,neuron_type,activation))
        
        if features != 2 and order != 1:
            raise ValueError("This model accept only features = 2 when order is set to 2")
        
        self.features = features
        if activation == "R":
            self.activation = ReLU
            self.activation_dx = ReLU_Deriv
            self.activation_name = "ReLU"
            
        if activation == "S":
            self.activation = Sigmoid
            self.activation_dx = Sigmoid_Deriv
            self.activation_name = "Sigmoid"
            
        if activation == "T":
            self.activation = TanH
            self.activation_dx = TanH_Deriv
            self.activation_name = "TanH"
            
        if activation == "L":
            self.activation = Linear
            self.activation_dx = Linear_Deriv
            self.activation_name = "Linear"
        
        self.order = order 
        self.neuron_type = neuron_type 
        self.reset()    
    
    def reset(self,RanDomRange = 0.5):
        self.loss = np.array([])
        self.val_loss = np.array([])
        
        self.acc = np.array([])
        self.val_acc = np.array([])
        
        self.miss_classified_train = np.array([])
        self.miss_classified_valid = np.array([])

        if self.order == 1 :
            self.weights = np.random.uniform(-RanDomRange,RanDomRange,3)

        elif self.order == 2 :
                self.weights = np.random.uniform(-RanDomRange,RanDomRange,6)
                
        self.weights_history = np.array([self.weights]) 
        
    def kernel(self,x):
        try :
            x.T[0].shape[0]
            x = x.T
        except:
            x = np.array([x])
            x = x.T
            
        if self.order == 1:
            return np.c_[ np.ones(x.T.shape[0]),x.T ].astype(np.float64)
        
        return np.array([[1]*x[0].shape[0], x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]]).astype(np.float64).T
            
        
    def fit(self,x_train,y_train,x_valid,y_valid,x_test,y_test,learning_rate,epoch):
        if x_train.shape[1] != self.features :
            raise ValueError("Passed features count is not equal to setted features count")
        
        self.reset()
        x_train, x_valid = self.kernel(x_train),self.kernel(x_valid) 
        
        for _ in range(epoch): 
            if self.neuron_type == "P":
                for index in range(x_train.shape[0]):               
                    predict = self.predict(np.array([x_train[index]]))[0]              
                    update = learning_rate*(y_train[index]-predict)
                    self.weights = self.weights + update*x_train[index]
                    
            elif self.neuron_type == "A":
                for index in range(x_train.shape[0]):
                    output = np.dot(x_train[index],self.weights)
                    errors = y_train[index] - output
                    self.weights += learning_rate*x_train[index]*errors  
            
            if self.neuron_type == "P":
                self.loss = np.append(self.loss,mean_squared_error(y_train,self.predict(x_train)))
                self.val_loss = np.append(self.val_loss,mean_squared_error(y_valid,self.predict(x_valid)))
            else:                
                self.loss = np.append(self.loss,mean_squared_error(y_train,np.dot(x_train,self.weights)))
                self.val_loss = np.append(self.val_loss,mean_squared_error(y_valid,np.dot(x_valid,self.weights)))
                
            self.acc = np.append(self.acc,accuracy_score(y_train,self.predict(x_train)))
            self.val_acc = np.append(self.val_acc,accuracy_score(y_valid,self.predict(x_valid)))      
            
            train_cm = confusion_matrix(y_train,self.predict(x_train))
            valid_cm = confusion_matrix(y_valid,self.predict(x_valid))
            self.miss_classified_train = np.append(self.miss_classified_train,sum([train_cm[0,1],train_cm[1,0]])/x_train.shape[0]) 
            self.miss_classified_valid = np.append(self.miss_classified_valid,sum([valid_cm[0,1],valid_cm[1,0]])/x_valid.shape[0]) 
                  
            self.weights_history = np.vstack([self.weights_history, self.weights])
            
        self.save_results(learning_rate,x_test,y_test)            
        

            
    def predict (self,x):
        if x.shape[1] != self.weights.shape[0]: 
            x = self.kernel(x)
            
        output = np.dot(x,self.weights.T)
        activate = np.array(list(map(self.activation,output)))  
        
        if self.activation_name == "ReLU" : return np.where(activate> 0, 1, -1)
        if self.activation_name == "Sigmoid" : return np.where(activate> 0.5, 1, -1)
        if self.activation_name == "TanH" : return np.where(activate> 0, 1, -1)
        if self.activation_name == "Linear" : return np.where(activate> 0, 1, -1)
        
        
    def save_results(self,learning_rate,x_test=None,y_test=None):
        if self.loss.shape[0] == 0 :
            return

        file_path = datetime.now().strftime("%Y-%m-%d %H-%M-%S") 
        path = ""
        
        if self.neuron_type == "P": path += "Perceptron - Order "
        else :                      path += "Adaline - Order "
        
        if self.order == 1 :  path += str(1)
        else :                path += str(2)
         
        path += " - " + self.activation_name
        path += " - LR " + str(learning_rate)         
        path += " - Iteration " + str(self.loss.shape[0])  
           
        file_path += " - "+ path           
        os.mkdir(file_path)
        
        with open(file_path + '/History.npy', 'wb') as f:
            np.save(f, self.weights)
            np.save(f, self.weights_history)            

        plt.plot(list(range(self.loss.shape[0])),self.loss,label="Training Loss",color="purple")
        plt.plot(list(range(self.val_loss.shape[0])),self.val_loss,label="Validation Loss",color="red")           
        plt.xlabel("Iteration"); plt.ylabel("Loss(MSE)")
        plt.title(path); plt.legend(); plt.savefig(file_path + '/loss.png'); plt.show() ;plt.clf()
        
        plt.plot(list(range(self.acc.shape[0])),self.acc,label="Training Accuracy",color="blue")
        plt.plot(list(range(self.val_acc.shape[0])),self.val_acc,label="Validation Accuracy",color="green")           
        plt.xlabel("Iteration"); plt.ylabel("Accuracy")
        plt.title(path); plt.legend(); plt.savefig(file_path + '/accuracy.png'); plt.show() ;plt.clf()        
        
        plt.plot(list(range(self.miss_classified_train.shape[0])),self.miss_classified_train,label="Training Misclassified",color="black")
        plt.plot(list(range(self.miss_classified_valid.shape[0])),self.miss_classified_valid,label="Validation Misclassified",color="gold")           
        plt.xlabel("Iteration"); plt.ylabel("Misclassified Rate")
        plt.title(path); plt.legend(); plt.savefig(file_path + '/Misclassified.png'); plt.show() ;plt.clf()
        
        if self.order != 1 :    
            cols = ["Bias","X1","X2","X1^2","X2^2","X1*X2","Label"]
            info = pd.DataFrame(np.append(self.kernel(x_test),y_test,axis=1),columns=cols)
            sns.pairplot(info,hue="Label"); plt.savefig(file_path + '/pairplot.png'); plt.show(); plt.clf()
        
        N =  500
        _range_x = np.linspace(int(x_test.T[0].min())-2,int(x_test.T[0].max())+2,N)
        _range_y = np.linspace(int(x_test.T[1].min())-2,int(x_test.T[1].max())+2,N)
        _range_y, _range_x = np.meshgrid(_range_x, _range_y)
        _range_z = self.predict(np.array([np.reshape(_range_x,(N*N,)),np.reshape(_range_y,(N*N,))]).T)
        _range_z = np.reshape(_range_z,(N,N))

        fig, ax = plt.subplots()
        cmap = ListedColormap(["darkorange", "lightseagreen"])            
        c = ax.pcolormesh(_range_x, _range_y, _range_z,cmap=cmap, vmin=-1.5, vmax=1.75)
        ax.set_title(path+" | Test Samples")
        ax.axis([_range_x.min(), _range_x.max(), _range_y.min(), _range_y.max()])
         
        if self.order == 1:
            _range = np.linspace(int(x_test.T[0].min())-1,int(x_test.T[0].max())+1,500)                
            for _ in range(0,self.weights_history.shape[0]):
                b,w1,w2 = self.weights_history[_]
                ax.plot(_range,[-tmp*w1/w2 - b/w2 for tmp in _range],color="yellow",alpha=0.2)
                    
            b,w1,w2 = self.weights[0],self.weights[1],self.weights[2]
            ax.plot(_range,[-tmp*w1/w2 - b/w2 for tmp in _range],color="green",linestyle = 'dashed',label="boundary")  
                  
            plt.xlabel("Feature - 1")
            plt.ylabel("Feature - 2")
            
        ax.scatter(x_test[(y_test==-1).T[0]].T[0],x_test[(y_test==-1).T[0]].T[1],color="darkorange",alpha=0.4,label="Class -1",edgecolors='black')
        ax.scatter(x_test[(y_test==1).T[0]].T[0],x_test[(y_test==1).T[0]].T[1],color="lightseagreen",alpha=0.4,label="Class 1",edgecolors='black')
        
        plt.legend(); plt.savefig(file_path + '/order-plot.png'); plt.show(); plt.clf()    

        
        if x_test is not None:
            CalculateMetricsAndPlot(y_test,self.predict(x_test),"Blues",path,file_path)        
    