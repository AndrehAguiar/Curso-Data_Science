import matplotlib.pyplot as plt
import pandas as pd

class Histogram(object):
    
    def __init__(self):
        
        self.ax = []
        
        
    def format_percent(self, x)->str:
        return "{0:.2f}%".format(x*100)


    def ticks_calculate(self, values:list, bars:int)->list:
        values.sort()
        ticks=[values[0]]
        for _ in range(bars):
            ticks.append(ticks[-1]+(values[-1]-values[0])/bars)
            
        return ticks # Retorna a lista com os valores dos ticks para o eixo x
    
    
    def relative_frequency(self, yticks:list, total:int)->list:
        return [self.format_percent(tick/total) for tick in yticks] # Retorna lista de labels(str) do ytick
    
    
    def plot_histogram(self, bars:int, data:pd.Series, title:str, unit:str)->plt.hist:
        ax = data.plot.hist(bins=bars, rwidth=0.95)

        ax.set_yticklabels(self.relative_frequency(ax.get_yticks(), len(data))) # envia lista de yticks e tamanho do lista de valores
        ax.set_xticks(self.ticks_calculate(data.values, bars)) # envia lista de valores e quantidade de barras

        fig = plt.gcf()
        fig.set_size_inches(10,5)
        fig.set_dpi(150)

        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_ylabel('Relative Frequency')
        ax.grid(axis='y')