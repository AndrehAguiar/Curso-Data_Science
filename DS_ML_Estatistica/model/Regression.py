import matplotlib.pyplot as plt

class Regression(object):
    
    def __init__(self, x:list=[], y:list=[]):
        self.x = x
        self.y = y
        
        
    def set_data(self, x:list, y:list):

        if len(x)!=len(y):
            return print('x and y must be the same size')
        
        self.x = x
        self.y = y
        self.m = self.get_m()
        self.b = self.get_b()
        self.r = self.get_r()
        self.sqr_r = self.get_r(True)
        self.score = self.get_score()
        self.rmse = self.get_rmse()
        

    def get_m(self)->float:

        n = len(self.x)

        return ((n * (self.x * self.y).sum()) - (self.x.sum() * self.y.sum()))\
        / ((n * (self.x**2).sum()) - ((self.x.sum())**2))


    def get_b(self)->float:
        return (self.y.sum() - self.get_m() * self.x.sum()) / len(self.x)


    def get_r(self, square:bool=False):

        if square:
            return (self.y - [self.get_predict(target) for target in self.x])**2

        else:
            return self.y - [self.get_predict(target) for target in self.x]


    def sum_sqr_r(self)->float:
        return self.get_r(True).sum()


    def get_predict(self, target:float)->float:
        return self.get_m() * target + self.get_b()


    def sum_sqr_total(self)->float:
        return ((self.y - self.y.mean())**2).sum()


    def get_score(self)->float:
        return 1 - self.sum_sqr_r() / self.sum_sqr_total()
    
    
    def get_rmse(self):
        return (self.sum_sqr_r()/len(self.x))**0.5
    
    
    def fit_prediction(self)->list:
        return [self.get_predict(target) for target in self.x]


    def plot_regression(self, title:str, labels:list, emp_rule:int=0)->plt.scatter:
        
        fig = plt.gcf()
        fig.set_size_inches(7,7)
        fig.set_dpi(150)
        
        plt.scatter(self.x, self.y, s=10, label=labels[0])
        plt.plot(self.x, [self.get_predict(target) for target in self.x],
                 color='y', label='Regression')
        
        
        if emp_rule > 0:
            plt.plot(self.x, self.fit_prediction()+1*self.rmse, color='yellow', label='(+/-) 1 Std. Deviation')
            plt.plot(self.x, self.fit_prediction()-1*self.rmse, color='yellow')
            
        if emp_rule > 1:

            plt.plot(self.x, self.fit_prediction()+2*self.rmse, color='orange', label='(+/-) 2 Std. Deviation')
            plt.plot(self.x, self.fit_prediction()-2*self.rmse, color='orange')
            
        if emp_rule > 2:

            plt.plot(self.x, self.fit_prediction()+3*self.rmse, color='red', label='(+/-) 3 Std. Deviation')
            plt.plot(self.x, self.fit_prediction()-3*self.rmse, color='red')
        
        
        plt.title(f'{title}\n{labels[0]} Vs {labels[1]}')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        
        plt.grid()
        plt.legend()
        
        plt.show()