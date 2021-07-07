# Ultrasound Variational Style Transfer to Generate Images Beyond the Observed Domain

Official code for paper [Ultrasound Variational Style Transfer to Generate Images Beyond the Observed Domain](google.com)

# Pretrained Model
The pretrained model for our network can be found [here](https://drive.google.com/file/d/15g-j4WSp74emhapwHXzyOSR_q8YDaVKO/view?usp=sharing), where you should initialize the network by the following parameters:
```python
	model=StyleTransfer(5,[64,128,256,512,1024],1,rec=0,
        p=[0,0,0,0,0.01],
        s=[0.1,0.002,0.001,0.01,10],
        tv=0,
        name='style_transfer')
    	inputs=tf.keras.Input(shape=(256,256,2,))
    	model(inputs)
	model.load_weights('27.h5',by_name=True,skip_mismatch=True)
```
# Credit 
If you use the code or the paper in any of your work, please remember to cite us
```bash
```