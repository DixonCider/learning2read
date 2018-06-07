from pyecharts.base import Base
import pandas as pd
import types
class PlotBase(Base): # my base :)
    def __init__(self,w=600,h=400,*args,**kwargs):
        super(PlotBase,self).__init__(width=w,height=h,*args,**kwargs)

class SimpleScatter(PlotBase): # give (data,x,y) -> scatter
    def setup_xAxis(self,atype="value"):
        self._option['xAxis']={'type': atype}
        return self
    def setup_yAxis(self,atype="value"):
        self._option['yAxis']={'type': atype}
        return self
    def setup_data(self,data):
        if isinstance(data,pd.DataFrame):
            self.list_of_dict=data.to_dict("record")
        elif isinstance(data,types.GeneratorType):
            self.list_of_dict=data
        else:
            self.list_of_dict=data
        return self
    def setup_dataZoom(self,use_slider):
        self._option['dataZoom']=[
            {'type':'inside','xAxisIndex':0,'filterMode':'empty'},
            {'type':'inside','yAxisIndex':0,'filterMode':'empty'}
        ]
        if use_slider:
            self._option['dataZoom'].append({'type':'slider','xAxisIndex':0,'filterMode':'empty'})
            self._option['dataZoom'].append({'type':'slider','yAxisIndex':0,'filterMode':'empty'})

    def setup(self,data,use_slider=False,**kwargs):
        self.setup_data(data)
        self.setup_xAxis()
        self.setup_yAxis()
        self._option['legend']=[{'data':[]}]
        self._option['series']=[]
        self._option['tooltip']={'show':True}
        self.setup_dataZoom(use_slider)
        return self
    def add(self,xname="x",yname="y",series_type="scatter",legend=None,fvalue=None,fsymbol=None,fsymbolSize=None,ftooltip=None,fitemstyle=None):
        fvalue = fvalue or (lambda r:[r[xname],r[yname]])
        fsymbol= fsymbol or (lambda r:"circle")
        fsymbolSize=fsymbolSize or (lambda r:5)
        ftooltip=ftooltip or (lambda r:{'formatter':'(%.4f,%.4f)'%(r[xname],r[yname])})
        fitemstyle=fitemstyle or (lambda r:{})
        new_series={}
        new_series['type']=series_type
        new_series['data']=[{
            'value':fvalue(r),
            'symbol':fsymbol(r),
            'symbolSize':fsymbolSize(r),
            'itemStyle':fitemstyle(r),
            'tooltip':ftooltip(r),
        } for r in self.list_of_dict]
        if legend:
            new_series['name']=legend
            self._option.get('legend')[0].get('data').append(legend)

        self._option.get('series').append(new_series)
        return self
    @classmethod
    def demo(cls):
        import math # ugly
        return cls().setup([
            {'x':t*.1-10,'y':math.sin(t*.1-10)} for t in range(200)
        ],True).add(ftooltip=lambda r:{'formatter':'sin(%.4f)=%.4f'%(r['x'],r['y'])})

class SimpleLine(SimpleScatter):
    def add(self,*args,**kwargs):
        return super(SimpleLine,self).add(series_type="line",*args,**kwargs)


class SimpleBar(SimpleScatter):
    def setup_xAxis(self,atype="category"):
        self._option['xAxis']={'type': atype}
    def add(self,*args,**kwargs):
        return super(SimpleBar,self).add(series_type="bar",*args,**kwargs)
    @classmethod
    def demo(cls):
        import random # ugly
        import math   # ugly
        return cls().setup([
            {'x':i,'y':random.random(),'z':random.random()/2+0.25}
        for i in range(20)]).add(legend="y").add(yname='z',legend="z")

class HeatMap(SimpleScatter):
    def setup_visualMap(self, min=0, max=1, splitNumber=5, opacity=0.66, fcolor=None):
        fcolor = fcolor or (lambda t:"#%02x%02x%02x"%(t*30+120,128,(4-t)*30+120))
        self._option['visualMap']={
            'type':'piecewise',
            'min':min,
            'max':max,
            'splitNumber':splitNumber,
            'inRange':{
                'opacity':opacity,
                'color':[fcolor(t) for t in range(splitNumber)],
            },
        }
    def setup(self,data,use_slider=False,**kwargs):
        self.setup_data(data)
        self.setup_xAxis('category')
        self.setup_yAxis('category')
        self._option['legend']=[{'data':[]}]
        self._option['series']=[]
        self._option['tooltip']={'show':True}
        self.setup_dataZoom(use_slider)
        return self
    def add(self,xname="x",yname="y",zname="z",fvalue=None,ftooltip=None):
        fvalue = fvalue or (lambda r:[r[xname],r[yname],r[zname]])
        ftooltip=ftooltip or (lambda r:{'formatter':'(%.4f,%.4f)-> p=%.4f'%(r[xname],r[yname],r[zname])})
        new_series={}
        new_series['type']="heatmap"
        new_series['data']=[{
            'value':fvalue(r),
            'tooltip':ftooltip(r),
        } for r in self.list_of_dict]
        new_series['itemStyle']={'emphasis':{
            'borderColor':'#333',
            'borderWidth':1
        }}
        self._option.get('series').append(new_series)
        return self
"""
# {-1,0,[1-10]}

obj._option['visualMap']={
    'type':'piecewise',
    'pieces': [
        {'gte' : 8, 'lte' : 10, 'color' : 'blue'},
        {'gte' : 5, 'lte' : 7, 'color' : 'yellow'},
        {'gte' : 1, 'lte' : 4, 'color' : 'red'},
        {'value' : -1, 'color' : 'black'},
        {'value' : 0, 'color' : 'grey'},
    ]
}
"""

"""
obj=DEV(800,800)
data=[]
N=39
m=1
for i in range(-N,N):
    for j in range(-N,N):
        x1=i*m
        x2=j*m
        data.append({'x1':x1, 'x2':x2 , 'p':origin_f(x1,x2) })
obj.setup(data)
obj.add("x1","x2","p")
obj
"""