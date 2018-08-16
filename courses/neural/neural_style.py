import importlib
import utils2;importlib.reload(utils2)
from scipy.optimize import fmin_l_bgfs_b
from scipy.misc import imsave
from keras import metrics
from vgg16_avg import VGG16_Avg

# allow_growth option,  attempts to allocate only as much GPU memory based on runtime allocations
limit_mem()

path = '--path to image here-'
fnames=glob.glob(path+'**/*.jpeg',recursive=True)
content_file=fnames[0]
style_file=fnames[1]

rn_mean=np.array([123.68, 116.779, 103.939],dtype=np.float32)
#preproc lambda
preproc= lambda x: (x-rn_mean)[:,:,:,::-1]

#deproc lambda
deproc= lambda x,s : np.clip(x.reshape(s)[:,:,:,::-1] +rn_mean,0,255)

#read image into img
content_img=Image.open(content_file)
content_img

# read style image
style_img=Image.open(style_file)
style_img

#create numpy array from image, preprocess and expand dimension
content_img_arr=preproc(np.expand_dims(np.array(content_img),0))

#resize content image to fit style image dimensions
s_h,s_w=style_img.size()

content_resized_arr=content_img_arr[:,:s_h,:s_w]

shp=content_img_arr.shp

style_img_arr=preproc(np.expand_dims(np.array(style_img),0))
style_shp=style_img_arr.shp


#create vgg 16 model
model=Vgg16_Avg(include_top=False)

# get the conv layer  output that we need for content
#layer model
content_model=model(vgg_model.input,model.getlayer("block5_conv_2").output)

# register a tensorflow variable for the target
content_target=K.variable(content_model.predict(content_resized_arr))

#create the style model- has one input but multiple outputs
style_outputs={ l.name:l.output  for l in style}
style_layers=[ style_outputs[f'block{}_conv2'.format(i)] for i in range(1,6)]
style_model=model(vgg_model.input,style_layers)

# register a tensorflow variable for the style target
style_target=K.variable(content_model.predict(content_resized_arr))

class Evaluator:
	def __init__(self,f,shp):
		self.f=f
		self.shp=shp
	# loss. input x has to be reshaped as we gave a flattened array
	def loss(self,x):
		loss,self.grad_values=self.f(x.reshape(self.shp))
		return loss.astype(np.float64)

	# gradient.  note we need to give a flattened array to lbgfs
	def gradients(self,x):	
		return self.grad_values.flatten().astype(np.float64)
		
loss= K.mean(metrics.mse(layer,target))
grads=K.gradients(loss,model.input)
function=K.function([model.input],[loss]+grads)
evaluator=Evaluator(function,shp)

def refine_image(no_iter,eval_obj,x):
	for i in range(no_iter):
		# here x is the initial guess
		x,min_val,info=fmin_l_bfgs_b(eval_obj.loss,x.flatten, fprime=eval_obj.gradients)
		x=np.clip(x,-127,127)
		print "current loss:"+min_val
		imsave(f'{path}/results/res_iter{i}',deproc(x.copy(),shp)[0])
	return x


rand_img= lambda shape: np.random.uniform(-2.5,2.5,shape)/100
x=rand_img(shp)
iters=10
refine_image(iters,evaluator,x)	
Image.open('{path}/results/res_iter9')
