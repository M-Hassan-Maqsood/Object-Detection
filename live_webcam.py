#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ultralytics')


# In[1]:


from ultralytics import YOLO


# In[ ]:


from ultralytics.models.yolo.detect.predict import DetectionPredictor


# In[2]:


model=YOLO('best full human.pt' ,task='detect')
model.predict(source='0', show=True, conf=0.5)

