��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
y

SegmentSum	
data"T
segment_ids"Tindices
output"T" 
Ttype:
2	"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8��
x
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*
shared_namelayer1/kernel
q
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel* 
_output_shapes
:
��
*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:
*
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

:
d*
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:d*
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d<*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:d<*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:<*
dtype0
v
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_namelayer4/kernel
o
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*
_output_shapes

:<*
dtype0
n
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer4/bias
g
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes
:*
dtype0
v
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer5/kernel
o
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*
_output_shapes

:*
dtype0
n
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer5/bias
g
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
X
ncVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namenc
Q
nc/Read/ReadVariableOpReadVariableOpnc*
_output_shapes
: *
dtype0
V
tVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namet
O
t/Read/ReadVariableOpReadVariableOpt*
_output_shapes
: *
dtype0
�
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*%
shared_nameAdam/layer1/kernel/m

(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m* 
_output_shapes
:
��
*
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
:
*
dtype0
�
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

:
d*
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:d*
dtype0
�
Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d<*%
shared_nameAdam/layer3/kernel/m
}
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*
_output_shapes

:d<*
dtype0
|
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*#
shared_nameAdam/layer3/bias/m
u
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes
:<*
dtype0
�
Adam/layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*%
shared_nameAdam/layer4/kernel/m
}
(Adam/layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/m*
_output_shapes

:<*
dtype0
|
Adam/layer4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer4/bias/m
u
&Adam/layer4/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/m*
_output_shapes
:*
dtype0
�
Adam/layer5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer5/kernel/m
}
(Adam/layer5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer5/bias/m
u
&Adam/layer5/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/m*
_output_shapes
:*
dtype0
�
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*%
shared_nameAdam/layer1/kernel/v

(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v* 
_output_shapes
:
��
*
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
:
*
dtype0
�
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
d*%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

:
d*
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:d*
dtype0
�
Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d<*%
shared_nameAdam/layer3/kernel/v
}
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*
_output_shapes

:d<*
dtype0
|
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*#
shared_nameAdam/layer3/bias/v
u
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes
:<*
dtype0
�
Adam/layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*%
shared_nameAdam/layer4/kernel/v
}
(Adam/layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/v*
_output_shapes

:<*
dtype0
|
Adam/layer4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer4/bias/v
u
&Adam/layer4/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer4/bias/v*
_output_shapes
:*
dtype0
�
Adam/layer5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer5/kernel/v
}
(Adam/layer5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer5/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer5/bias/v
u
&Adam/layer5/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�7B�7 B�7
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem]m^m_m`mambmcmd$me%mfvgvhvivjvkvlvmvn$vo%vp
 
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
�
/non_trainable_variables
0layer_regularization_losses
regularization_losses
trainable_variables
1layer_metrics

2layers
		variables
3metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
4non_trainable_variables
	variables
regularization_losses
trainable_variables
5layer_metrics

6layers
7layer_regularization_losses
8metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
9non_trainable_variables
	variables
regularization_losses
trainable_variables
:layer_metrics

;layers
<layer_regularization_losses
=metrics
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
>non_trainable_variables
	variables
regularization_losses
trainable_variables
?layer_metrics

@layers
Alayer_regularization_losses
Bmetrics
YW
VARIABLE_VALUElayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
Cnon_trainable_variables
 	variables
!regularization_losses
"trainable_variables
Dlayer_metrics

Elayers
Flayer_regularization_losses
Gmetrics
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
�
Hnon_trainable_variables
&	variables
'regularization_losses
(trainable_variables
Ilayer_metrics

Jlayers
Klayer_regularization_losses
Lmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
#
0
1
2
3
4

M0
N1
O2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ptotal
	Qcount
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
G
Ync
Y	n_correct
Zt
	Ztotal
[	variables
\	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
IG
VARIABLE_VALUEnc1keras_api/metrics/2/nc/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEt0keras_api/metrics/2/t/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

[	variables
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
s
serving_default_args_0_1Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
a
serving_default_args_0_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_17973
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpnc/Read/ReadVariableOpt/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer3/kernel/m/Read/ReadVariableOp&Adam/layer3/bias/m/Read/ReadVariableOp(Adam/layer4/kernel/m/Read/ReadVariableOp&Adam/layer4/bias/m/Read/ReadVariableOp(Adam/layer5/kernel/m/Read/ReadVariableOp&Adam/layer5/bias/m/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer3/kernel/v/Read/ReadVariableOp&Adam/layer3/bias/v/Read/ReadVariableOp(Adam/layer4/kernel/v/Read/ReadVariableOp&Adam/layer4/bias/v/Read/ReadVariableOp(Adam/layer5/kernel/v/Read/ReadVariableOp&Adam/layer5/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_18478
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1nctAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer3/kernel/mAdam/layer3/bias/mAdam/layer4/kernel/mAdam/layer4/bias/mAdam/layer5/kernel/mAdam/layer5/bias/mAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer3/kernel/vAdam/layer3/bias/vAdam/layer4/kernel/vAdam/layer4/bias/vAdam/layer5/kernel/vAdam/layer5/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_18611��
�
�
&__inference_layer4_layer_call_fn_18301

inputs
unknown:<
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_176562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_18156

inputs	
inputs_1
inputs_2	
unknown:
��

	unknown_0:

	unknown_1:
d
	unknown_2:d
	unknown_3:d<
	unknown_4:<
	unknown_5:<
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_176852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�	
�
&__inference_layer1_layer_call_fn_18241

inputs	
inputs_1
inputs_2	
unknown:
��

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2unknown	unknown_0*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_176052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������:: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�;
�
A__inference_layer1_layer_call_and_return_conditional_losses_18230

inputs	
inputs_1
inputs_2	B
.embedding_lookup_sparse_embedding_lookup_18204:
��
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�(embedding_lookup_sparse/embedding_lookup{
SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SparseFillEmptyRows/Const�
'SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsinputsinputs_1inputs_2"SparseFillEmptyRows/Const:output:0*
T0*T
_output_shapesB
@:���������:���������:���������:���������2)
'SparseFillEmptyRows/SparseFillEmptyRows{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSlice8SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
+embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+embedding_lookup_sparse/strided_slice/stack�
-embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-embedding_lookup_sparse/strided_slice/stack_1�
-embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-embedding_lookup_sparse/strided_slice/stack_2�
%embedding_lookup_sparse/strided_sliceStridedSlice8SparseFillEmptyRows/SparseFillEmptyRows:output_indices:04embedding_lookup_sparse/strided_slice/stack:output:06embedding_lookup_sparse/strided_slice/stack_1:output:06embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2'
%embedding_lookup_sparse/strided_slice�
embedding_lookup_sparse/UniqueUniquestrided_slice:output:0*
T0	*2
_output_shapes 
:���������:���������2 
embedding_lookup_sparse/Unique�
(embedding_lookup_sparse/embedding_lookupResourceGather.embedding_lookup_sparse_embedding_lookup_18204"embedding_lookup_sparse/Unique:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@embedding_lookup_sparse/embedding_lookup/18204*'
_output_shapes
:���������
*
dtype02*
(embedding_lookup_sparse/embedding_lookup�
1embedding_lookup_sparse/embedding_lookup/IdentityIdentity1embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding_lookup_sparse/embedding_lookup/18204*'
_output_shapes
:���������
23
1embedding_lookup_sparse/embedding_lookup/Identity�
3embedding_lookup_sparse/embedding_lookup/Identity_1Identity:embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
25
3embedding_lookup_sparse/embedding_lookup/Identity_1�
embedding_lookup_sparse/CastCast.embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
embedding_lookup_sparse/Cast�
%embedding_lookup_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%embedding_lookup_sparse/GatherV2/axis�
 embedding_lookup_sparse/GatherV2GatherV2<embedding_lookup_sparse/embedding_lookup/Identity_1:output:0$embedding_lookup_sparse/Unique:idx:0.embedding_lookup_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������
2"
 embedding_lookup_sparse/GatherV2~
embedding_lookup_sparse/RankConst*
_output_shapes
: *
dtype0*
value	B :2
embedding_lookup_sparse/Rank�
embedding_lookup_sparse/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
embedding_lookup_sparse/sub/y�
embedding_lookup_sparse/subSub%embedding_lookup_sparse/Rank:output:0&embedding_lookup_sparse/sub/y:output:0*
T0*
_output_shapes
: 2
embedding_lookup_sparse/sub�
&embedding_lookup_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&embedding_lookup_sparse/ExpandDims/dim�
"embedding_lookup_sparse/ExpandDims
ExpandDimsembedding_lookup_sparse/sub:z:0/embedding_lookup_sparse/ExpandDims/dim:output:0*
T0*
_output_shapes
:2$
"embedding_lookup_sparse/ExpandDims�
"embedding_lookup_sparse/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"embedding_lookup_sparse/ones/Const�
embedding_lookup_sparse/onesFill+embedding_lookup_sparse/ExpandDims:output:0+embedding_lookup_sparse/ones/Const:output:0*
T0*
_output_shapes
:2
embedding_lookup_sparse/ones�
embedding_lookup_sparse/ShapeShape7SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0*
_output_shapes
:2
embedding_lookup_sparse/Shape�
#embedding_lookup_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#embedding_lookup_sparse/concat/axis�
embedding_lookup_sparse/concatConcatV2&embedding_lookup_sparse/Shape:output:0%embedding_lookup_sparse/ones:output:0,embedding_lookup_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
embedding_lookup_sparse/concat�
embedding_lookup_sparse/ReshapeReshape7SparseFillEmptyRows/SparseFillEmptyRows:output_values:0'embedding_lookup_sparse/concat:output:0*
T0*'
_output_shapes
:���������2!
embedding_lookup_sparse/Reshape�
embedding_lookup_sparse/mulMul)embedding_lookup_sparse/GatherV2:output:0(embedding_lookup_sparse/Reshape:output:0*
T0*'
_output_shapes
:���������
2
embedding_lookup_sparse/mul�
embedding_lookup_sparse
SegmentSumembedding_lookup_sparse/mul:z:0 embedding_lookup_sparse/Cast:y:0*
T0*
Tindices0*'
_output_shapes
:���������
2
embedding_lookup_sparse�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd embedding_lookup_sparse:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp)^embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2T
(embedding_lookup_sparse/embedding_lookup(embedding_lookup_sparse/embedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�

�
A__inference_layer4_layer_call_and_return_conditional_losses_18292

inputs0
matmul_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_17823

inputs	
inputs_1
inputs_2	 
layer1_17796:
��

layer1_17798:

layer2_17801:
d
layer2_17803:d
layer3_17806:d<
layer3_17808:<
layer4_17811:<
layer4_17813:
layer5_17816:
layer5_17818:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2layer1_17796layer1_17798*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_176052 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_17801layer2_17803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_176222 
layer2/StatefulPartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_17806layer3_17808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_176392 
layer3/StatefulPartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_17811layer4_17813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_176562 
layer4/StatefulPartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_17816layer5_17818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_176772 
layer5/StatefulPartitionedCall�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const�
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�

�
A__inference_layer3_layer_call_and_return_conditional_losses_17639

inputs0
matmul_readvariableop_resource:d<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
A__inference_layer5_layer_call_and_return_conditional_losses_18316

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
SigmoidS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A2
mul/y`
mulMulSigmoid:y:0mul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2mul:z:0add/y:output:0*
T0*'
_output_shapes
:���������2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_17685

inputs	
inputs_1
inputs_2	 
layer1_17606:
��

layer1_17608:

layer2_17623:
d
layer2_17625:d
layer3_17640:d<
layer3_17642:<
layer4_17657:<
layer4_17659:
layer5_17678:
layer5_17680:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2layer1_17606layer1_17608*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_176052 
layer1/StatefulPartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_17623layer2_17625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_176222 
layer2/StatefulPartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_17640layer3_17642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_176392 
layer3/StatefulPartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_17657layer4_17659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_176562 
layer4/StatefulPartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_17678layer5_17680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_176772 
layer5/StatefulPartitionedCall�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Const�
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�

�
A__inference_layer4_layer_call_and_return_conditional_losses_17656

inputs0
matmul_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������<
 
_user_specified_nameinputs
�
�
&__inference_layer5_layer_call_fn_18325

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_176772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_layer2_layer_call_and_return_conditional_losses_17622

inputs0
matmul_readvariableop_resource:
d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�h
�
E__inference_sequential_layer_call_and_return_conditional_losses_18051

inputs	
inputs_1
inputs_2	I
5layer1_embedding_lookup_sparse_embedding_lookup_17993:
��
4
&layer1_biasadd_readvariableop_resource:
7
%layer2_matmul_readvariableop_resource:
d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:d<4
&layer3_biasadd_readvariableop_resource:<7
%layer4_matmul_readvariableop_resource:<4
&layer4_biasadd_readvariableop_resource:7
%layer5_matmul_readvariableop_resource:4
&layer5_biasadd_readvariableop_resource:
identity��layer1/BiasAdd/ReadVariableOp�/layer1/embedding_lookup_sparse/embedding_lookup�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/MatMul/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/MatMul/ReadVariableOp�layer5/BiasAdd/ReadVariableOp�layer5/MatMul/ReadVariableOp�
 layer1/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 layer1/SparseFillEmptyRows/Const�
.layer1/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsinputsinputs_1inputs_2)layer1/SparseFillEmptyRows/Const:output:0*
T0*T
_output_shapesB
@:���������:���������:���������:���������20
.layer1/SparseFillEmptyRows/SparseFillEmptyRows�
layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
layer1/strided_slice/stack�
layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
layer1/strided_slice/stack_1�
layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
layer1/strided_slice/stack_2�
layer1/strided_sliceStridedSlice?layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0#layer1/strided_slice/stack:output:0%layer1/strided_slice/stack_1:output:0%layer1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
layer1/strided_slice�
2layer1/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2layer1/embedding_lookup_sparse/strided_slice/stack�
4layer1/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4layer1/embedding_lookup_sparse/strided_slice/stack_1�
4layer1/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4layer1/embedding_lookup_sparse/strided_slice/stack_2�
,layer1/embedding_lookup_sparse/strided_sliceStridedSlice?layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0;layer1/embedding_lookup_sparse/strided_slice/stack:output:0=layer1/embedding_lookup_sparse/strided_slice/stack_1:output:0=layer1/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2.
,layer1/embedding_lookup_sparse/strided_slice�
%layer1/embedding_lookup_sparse/UniqueUniquelayer1/strided_slice:output:0*
T0	*2
_output_shapes 
:���������:���������2'
%layer1/embedding_lookup_sparse/Unique�
/layer1/embedding_lookup_sparse/embedding_lookupResourceGather5layer1_embedding_lookup_sparse_embedding_lookup_17993)layer1/embedding_lookup_sparse/Unique:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*H
_class>
<:loc:@layer1/embedding_lookup_sparse/embedding_lookup/17993*'
_output_shapes
:���������
*
dtype021
/layer1/embedding_lookup_sparse/embedding_lookup�
8layer1/embedding_lookup_sparse/embedding_lookup/IdentityIdentity8layer1/embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@layer1/embedding_lookup_sparse/embedding_lookup/17993*'
_output_shapes
:���������
2:
8layer1/embedding_lookup_sparse/embedding_lookup/Identity�
:layer1/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityAlayer1/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2<
:layer1/embedding_lookup_sparse/embedding_lookup/Identity_1�
#layer1/embedding_lookup_sparse/CastCast5layer1/embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2%
#layer1/embedding_lookup_sparse/Cast�
,layer1/embedding_lookup_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,layer1/embedding_lookup_sparse/GatherV2/axis�
'layer1/embedding_lookup_sparse/GatherV2GatherV2Clayer1/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0+layer1/embedding_lookup_sparse/Unique:idx:05layer1/embedding_lookup_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������
2)
'layer1/embedding_lookup_sparse/GatherV2�
#layer1/embedding_lookup_sparse/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#layer1/embedding_lookup_sparse/Rank�
$layer1/embedding_lookup_sparse/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$layer1/embedding_lookup_sparse/sub/y�
"layer1/embedding_lookup_sparse/subSub,layer1/embedding_lookup_sparse/Rank:output:0-layer1/embedding_lookup_sparse/sub/y:output:0*
T0*
_output_shapes
: 2$
"layer1/embedding_lookup_sparse/sub�
-layer1/embedding_lookup_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-layer1/embedding_lookup_sparse/ExpandDims/dim�
)layer1/embedding_lookup_sparse/ExpandDims
ExpandDims&layer1/embedding_lookup_sparse/sub:z:06layer1/embedding_lookup_sparse/ExpandDims/dim:output:0*
T0*
_output_shapes
:2+
)layer1/embedding_lookup_sparse/ExpandDims�
)layer1/embedding_lookup_sparse/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)layer1/embedding_lookup_sparse/ones/Const�
#layer1/embedding_lookup_sparse/onesFill2layer1/embedding_lookup_sparse/ExpandDims:output:02layer1/embedding_lookup_sparse/ones/Const:output:0*
T0*
_output_shapes
:2%
#layer1/embedding_lookup_sparse/ones�
$layer1/embedding_lookup_sparse/ShapeShape>layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0*
_output_shapes
:2&
$layer1/embedding_lookup_sparse/Shape�
*layer1/embedding_lookup_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*layer1/embedding_lookup_sparse/concat/axis�
%layer1/embedding_lookup_sparse/concatConcatV2-layer1/embedding_lookup_sparse/Shape:output:0,layer1/embedding_lookup_sparse/ones:output:03layer1/embedding_lookup_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%layer1/embedding_lookup_sparse/concat�
&layer1/embedding_lookup_sparse/ReshapeReshape>layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0.layer1/embedding_lookup_sparse/concat:output:0*
T0*'
_output_shapes
:���������2(
&layer1/embedding_lookup_sparse/Reshape�
"layer1/embedding_lookup_sparse/mulMul0layer1/embedding_lookup_sparse/GatherV2:output:0/layer1/embedding_lookup_sparse/Reshape:output:0*
T0*'
_output_shapes
:���������
2$
"layer1/embedding_lookup_sparse/mul�
layer1/embedding_lookup_sparse
SegmentSum&layer1/embedding_lookup_sparse/mul:z:0'layer1/embedding_lookup_sparse/Cast:y:0*
T0*
Tindices0*'
_output_shapes
:���������
2 
layer1/embedding_lookup_sparse�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAdd'layer1/embedding_lookup_sparse:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
layer1/BiasAdd�
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
layer2/MatMul/ReadVariableOp�
layer2/MatMulMatMullayer1/BiasAdd:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
layer2/MatMul�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
layer2/Relu�
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:d<*
dtype02
layer3/MatMul/ReadVariableOp�
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
layer3/MatMul�
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
layer3/BiasAdd/ReadVariableOp�
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
layer3/Relu�
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02
layer4/MatMul/ReadVariableOp�
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer4/MatMul�
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer4/BiasAdd/ReadVariableOp�
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer4/Relu�
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer5/MatMul/ReadVariableOp�
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer5/MatMul�
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer5/BiasAdd/ReadVariableOp�
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer5/BiasAddv
layer5/SigmoidSigmoidlayer5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer5/Sigmoida
layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A2
layer5/mul/y|

layer5/mulMullayer5/Sigmoid:y:0layer5/mul/y:output:0*
T0*'
_output_shapes
:���������2

layer5/mula
layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer5/add/yz

layer5/addAddV2layer5/mul:z:0layer5/add/y:output:0*
T0*'
_output_shapes
:���������2

layer5/add�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Consti
IdentityIdentitylayer5/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^layer1/BiasAdd/ReadVariableOp0^layer1/embedding_lookup_sparse/embedding_lookup^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2b
/layer1/embedding_lookup_sparse/embedding_lookup/layer1/embedding_lookup_sparse/embedding_lookup2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�T
�
__inference__traced_save_18478
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop!
savev2_nc_read_readvariableop 
savev2_t_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop3
/savev2_adam_layer4_kernel_m_read_readvariableop1
-savev2_adam_layer4_bias_m_read_readvariableop3
/savev2_adam_layer5_kernel_m_read_readvariableop1
-savev2_adam_layer5_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop3
/savev2_adam_layer4_kernel_v_read_readvariableop1
-savev2_adam_layer4_bias_v_read_readvariableop3
/savev2_adam_layer5_kernel_v_read_readvariableop1
-savev2_adam_layer5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/2/nc/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/2/t/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_nc_read_readvariableopsavev2_t_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop/savev2_adam_layer4_kernel_m_read_readvariableop-savev2_adam_layer4_bias_m_read_readvariableop/savev2_adam_layer5_kernel_m_read_readvariableop-savev2_adam_layer5_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop/savev2_adam_layer4_kernel_v_read_readvariableop-savev2_adam_layer4_bias_v_read_readvariableop/savev2_adam_layer5_kernel_v_read_readvariableop-savev2_adam_layer5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��
:
:
d:d:d<:<:<:::: : : : : : : : : : : :
��
:
:
d:d:d<:<:<::::
��
:
:
d:d:d<:<:<:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��
: 

_output_shapes
:
:$ 

_output_shapes

:
d: 

_output_shapes
:d:$ 

_output_shapes

:d<: 

_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��
: 

_output_shapes
:
:$ 

_output_shapes

:
d: 

_output_shapes
:d:$ 

_output_shapes

:d<: 

_output_shapes
:<:$ 

_output_shapes

:<: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::& "
 
_output_shapes
:
��
: !

_output_shapes
:
:$" 

_output_shapes

:
d: #

_output_shapes
:d:$$ 

_output_shapes

:d<: %

_output_shapes
:<:$& 

_output_shapes

:<: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::*

_output_shapes
: 
�h
�
E__inference_sequential_layer_call_and_return_conditional_losses_18129

inputs	
inputs_1
inputs_2	I
5layer1_embedding_lookup_sparse_embedding_lookup_18071:
��
4
&layer1_biasadd_readvariableop_resource:
7
%layer2_matmul_readvariableop_resource:
d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:d<4
&layer3_biasadd_readvariableop_resource:<7
%layer4_matmul_readvariableop_resource:<4
&layer4_biasadd_readvariableop_resource:7
%layer5_matmul_readvariableop_resource:4
&layer5_biasadd_readvariableop_resource:
identity��layer1/BiasAdd/ReadVariableOp�/layer1/embedding_lookup_sparse/embedding_lookup�layer2/BiasAdd/ReadVariableOp�layer2/MatMul/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/MatMul/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/MatMul/ReadVariableOp�layer5/BiasAdd/ReadVariableOp�layer5/MatMul/ReadVariableOp�
 layer1/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 layer1/SparseFillEmptyRows/Const�
.layer1/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsinputsinputs_1inputs_2)layer1/SparseFillEmptyRows/Const:output:0*
T0*T
_output_shapesB
@:���������:���������:���������:���������20
.layer1/SparseFillEmptyRows/SparseFillEmptyRows�
layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
layer1/strided_slice/stack�
layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
layer1/strided_slice/stack_1�
layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
layer1/strided_slice/stack_2�
layer1/strided_sliceStridedSlice?layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0#layer1/strided_slice/stack:output:0%layer1/strided_slice/stack_1:output:0%layer1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
layer1/strided_slice�
2layer1/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2layer1/embedding_lookup_sparse/strided_slice/stack�
4layer1/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4layer1/embedding_lookup_sparse/strided_slice/stack_1�
4layer1/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4layer1/embedding_lookup_sparse/strided_slice/stack_2�
,layer1/embedding_lookup_sparse/strided_sliceStridedSlice?layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0;layer1/embedding_lookup_sparse/strided_slice/stack:output:0=layer1/embedding_lookup_sparse/strided_slice/stack_1:output:0=layer1/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2.
,layer1/embedding_lookup_sparse/strided_slice�
%layer1/embedding_lookup_sparse/UniqueUniquelayer1/strided_slice:output:0*
T0	*2
_output_shapes 
:���������:���������2'
%layer1/embedding_lookup_sparse/Unique�
/layer1/embedding_lookup_sparse/embedding_lookupResourceGather5layer1_embedding_lookup_sparse_embedding_lookup_18071)layer1/embedding_lookup_sparse/Unique:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*H
_class>
<:loc:@layer1/embedding_lookup_sparse/embedding_lookup/18071*'
_output_shapes
:���������
*
dtype021
/layer1/embedding_lookup_sparse/embedding_lookup�
8layer1/embedding_lookup_sparse/embedding_lookup/IdentityIdentity8layer1/embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@layer1/embedding_lookup_sparse/embedding_lookup/18071*'
_output_shapes
:���������
2:
8layer1/embedding_lookup_sparse/embedding_lookup/Identity�
:layer1/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityAlayer1/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2<
:layer1/embedding_lookup_sparse/embedding_lookup/Identity_1�
#layer1/embedding_lookup_sparse/CastCast5layer1/embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2%
#layer1/embedding_lookup_sparse/Cast�
,layer1/embedding_lookup_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,layer1/embedding_lookup_sparse/GatherV2/axis�
'layer1/embedding_lookup_sparse/GatherV2GatherV2Clayer1/embedding_lookup_sparse/embedding_lookup/Identity_1:output:0+layer1/embedding_lookup_sparse/Unique:idx:05layer1/embedding_lookup_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������
2)
'layer1/embedding_lookup_sparse/GatherV2�
#layer1/embedding_lookup_sparse/RankConst*
_output_shapes
: *
dtype0*
value	B :2%
#layer1/embedding_lookup_sparse/Rank�
$layer1/embedding_lookup_sparse/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$layer1/embedding_lookup_sparse/sub/y�
"layer1/embedding_lookup_sparse/subSub,layer1/embedding_lookup_sparse/Rank:output:0-layer1/embedding_lookup_sparse/sub/y:output:0*
T0*
_output_shapes
: 2$
"layer1/embedding_lookup_sparse/sub�
-layer1/embedding_lookup_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-layer1/embedding_lookup_sparse/ExpandDims/dim�
)layer1/embedding_lookup_sparse/ExpandDims
ExpandDims&layer1/embedding_lookup_sparse/sub:z:06layer1/embedding_lookup_sparse/ExpandDims/dim:output:0*
T0*
_output_shapes
:2+
)layer1/embedding_lookup_sparse/ExpandDims�
)layer1/embedding_lookup_sparse/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)layer1/embedding_lookup_sparse/ones/Const�
#layer1/embedding_lookup_sparse/onesFill2layer1/embedding_lookup_sparse/ExpandDims:output:02layer1/embedding_lookup_sparse/ones/Const:output:0*
T0*
_output_shapes
:2%
#layer1/embedding_lookup_sparse/ones�
$layer1/embedding_lookup_sparse/ShapeShape>layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0*
_output_shapes
:2&
$layer1/embedding_lookup_sparse/Shape�
*layer1/embedding_lookup_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*layer1/embedding_lookup_sparse/concat/axis�
%layer1/embedding_lookup_sparse/concatConcatV2-layer1/embedding_lookup_sparse/Shape:output:0,layer1/embedding_lookup_sparse/ones:output:03layer1/embedding_lookup_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%layer1/embedding_lookup_sparse/concat�
&layer1/embedding_lookup_sparse/ReshapeReshape>layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0.layer1/embedding_lookup_sparse/concat:output:0*
T0*'
_output_shapes
:���������2(
&layer1/embedding_lookup_sparse/Reshape�
"layer1/embedding_lookup_sparse/mulMul0layer1/embedding_lookup_sparse/GatherV2:output:0/layer1/embedding_lookup_sparse/Reshape:output:0*
T0*'
_output_shapes
:���������
2$
"layer1/embedding_lookup_sparse/mul�
layer1/embedding_lookup_sparse
SegmentSum&layer1/embedding_lookup_sparse/mul:z:0'layer1/embedding_lookup_sparse/Cast:y:0*
T0*
Tindices0*'
_output_shapes
:���������
2 
layer1/embedding_lookup_sparse�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAdd'layer1/embedding_lookup_sparse:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
layer1/BiasAdd�
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
layer2/MatMul/ReadVariableOp�
layer2/MatMulMatMullayer1/BiasAdd:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
layer2/MatMul�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
layer2/Relu�
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:d<*
dtype02
layer3/MatMul/ReadVariableOp�
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
layer3/MatMul�
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
layer3/BiasAdd/ReadVariableOp�
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
layer3/Relu�
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02
layer4/MatMul/ReadVariableOp�
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer4/MatMul�
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer4/BiasAdd/ReadVariableOp�
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer4/Relu�
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer5/MatMul/ReadVariableOp�
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer5/MatMul�
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer5/BiasAdd/ReadVariableOp�
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer5/BiasAddv
layer5/SigmoidSigmoidlayer5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer5/Sigmoida
layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A2
layer5/mul/y|

layer5/mulMullayer5/Sigmoid:y:0layer5/mul/y:output:0*
T0*'
_output_shapes
:���������2

layer5/mula
layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer5/add/yz

layer5/addAddV2layer5/mul:z:0layer5/add/y:output:0*
T0*'
_output_shapes
:���������2

layer5/add�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Consti
IdentityIdentitylayer5/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^layer1/BiasAdd/ReadVariableOp0^layer1/embedding_lookup_sparse/embedding_lookup^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2b
/layer1/embedding_lookup_sparse/embedding_lookup/layer1/embedding_lookup_sparse/embedding_lookup2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
&__inference_layer3_layer_call_fn_18281

inputs
unknown:d<
	unknown_0:<
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_176392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�{
�	
 __inference__wrapped_model_17548

args_0	
args_0_1
args_0_2	T
@sequential_layer1_embedding_lookup_sparse_embedding_lookup_17491:
��
?
1sequential_layer1_biasadd_readvariableop_resource:
B
0sequential_layer2_matmul_readvariableop_resource:
d?
1sequential_layer2_biasadd_readvariableop_resource:dB
0sequential_layer3_matmul_readvariableop_resource:d<?
1sequential_layer3_biasadd_readvariableop_resource:<B
0sequential_layer4_matmul_readvariableop_resource:<?
1sequential_layer4_biasadd_readvariableop_resource:B
0sequential_layer5_matmul_readvariableop_resource:?
1sequential_layer5_biasadd_readvariableop_resource:
identity��(sequential/layer1/BiasAdd/ReadVariableOp�:sequential/layer1/embedding_lookup_sparse/embedding_lookup�(sequential/layer2/BiasAdd/ReadVariableOp�'sequential/layer2/MatMul/ReadVariableOp�(sequential/layer3/BiasAdd/ReadVariableOp�'sequential/layer3/MatMul/ReadVariableOp�(sequential/layer4/BiasAdd/ReadVariableOp�'sequential/layer4/MatMul/ReadVariableOp�(sequential/layer5/BiasAdd/ReadVariableOp�'sequential/layer5/MatMul/ReadVariableOp�
+sequential/layer1/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/layer1/SparseFillEmptyRows/Const�
9sequential/layer1/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsargs_0args_0_1args_0_24sequential/layer1/SparseFillEmptyRows/Const:output:0*
T0*T
_output_shapesB
@:���������:���������:���������:���������2;
9sequential/layer1/SparseFillEmptyRows/SparseFillEmptyRows�
%sequential/layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential/layer1/strided_slice/stack�
'sequential/layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'sequential/layer1/strided_slice/stack_1�
'sequential/layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'sequential/layer1/strided_slice/stack_2�
sequential/layer1/strided_sliceStridedSliceJsequential/layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0.sequential/layer1/strided_slice/stack:output:00sequential/layer1/strided_slice/stack_1:output:00sequential/layer1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2!
sequential/layer1/strided_slice�
=sequential/layer1/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential/layer1/embedding_lookup_sparse/strided_slice/stack�
?sequential/layer1/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2A
?sequential/layer1/embedding_lookup_sparse/strided_slice/stack_1�
?sequential/layer1/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2A
?sequential/layer1/embedding_lookup_sparse/strided_slice/stack_2�
7sequential/layer1/embedding_lookup_sparse/strided_sliceStridedSliceJsequential/layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0Fsequential/layer1/embedding_lookup_sparse/strided_slice/stack:output:0Hsequential/layer1/embedding_lookup_sparse/strided_slice/stack_1:output:0Hsequential/layer1/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask29
7sequential/layer1/embedding_lookup_sparse/strided_slice�
0sequential/layer1/embedding_lookup_sparse/UniqueUnique(sequential/layer1/strided_slice:output:0*
T0	*2
_output_shapes 
:���������:���������22
0sequential/layer1/embedding_lookup_sparse/Unique�
:sequential/layer1/embedding_lookup_sparse/embedding_lookupResourceGather@sequential_layer1_embedding_lookup_sparse_embedding_lookup_174914sequential/layer1/embedding_lookup_sparse/Unique:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*S
_classI
GEloc:@sequential/layer1/embedding_lookup_sparse/embedding_lookup/17491*'
_output_shapes
:���������
*
dtype02<
:sequential/layer1/embedding_lookup_sparse/embedding_lookup�
Csequential/layer1/embedding_lookup_sparse/embedding_lookup/IdentityIdentityCsequential/layer1/embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@sequential/layer1/embedding_lookup_sparse/embedding_lookup/17491*'
_output_shapes
:���������
2E
Csequential/layer1/embedding_lookup_sparse/embedding_lookup/Identity�
Esequential/layer1/embedding_lookup_sparse/embedding_lookup/Identity_1IdentityLsequential/layer1/embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
2G
Esequential/layer1/embedding_lookup_sparse/embedding_lookup/Identity_1�
.sequential/layer1/embedding_lookup_sparse/CastCast@sequential/layer1/embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������20
.sequential/layer1/embedding_lookup_sparse/Cast�
7sequential/layer1/embedding_lookup_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential/layer1/embedding_lookup_sparse/GatherV2/axis�
2sequential/layer1/embedding_lookup_sparse/GatherV2GatherV2Nsequential/layer1/embedding_lookup_sparse/embedding_lookup/Identity_1:output:06sequential/layer1/embedding_lookup_sparse/Unique:idx:0@sequential/layer1/embedding_lookup_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������
24
2sequential/layer1/embedding_lookup_sparse/GatherV2�
.sequential/layer1/embedding_lookup_sparse/RankConst*
_output_shapes
: *
dtype0*
value	B :20
.sequential/layer1/embedding_lookup_sparse/Rank�
/sequential/layer1/embedding_lookup_sparse/sub/yConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential/layer1/embedding_lookup_sparse/sub/y�
-sequential/layer1/embedding_lookup_sparse/subSub7sequential/layer1/embedding_lookup_sparse/Rank:output:08sequential/layer1/embedding_lookup_sparse/sub/y:output:0*
T0*
_output_shapes
: 2/
-sequential/layer1/embedding_lookup_sparse/sub�
8sequential/layer1/embedding_lookup_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8sequential/layer1/embedding_lookup_sparse/ExpandDims/dim�
4sequential/layer1/embedding_lookup_sparse/ExpandDims
ExpandDims1sequential/layer1/embedding_lookup_sparse/sub:z:0Asequential/layer1/embedding_lookup_sparse/ExpandDims/dim:output:0*
T0*
_output_shapes
:26
4sequential/layer1/embedding_lookup_sparse/ExpandDims�
4sequential/layer1/embedding_lookup_sparse/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential/layer1/embedding_lookup_sparse/ones/Const�
.sequential/layer1/embedding_lookup_sparse/onesFill=sequential/layer1/embedding_lookup_sparse/ExpandDims:output:0=sequential/layer1/embedding_lookup_sparse/ones/Const:output:0*
T0*
_output_shapes
:20
.sequential/layer1/embedding_lookup_sparse/ones�
/sequential/layer1/embedding_lookup_sparse/ShapeShapeIsequential/layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0*
_output_shapes
:21
/sequential/layer1/embedding_lookup_sparse/Shape�
5sequential/layer1/embedding_lookup_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5sequential/layer1/embedding_lookup_sparse/concat/axis�
0sequential/layer1/embedding_lookup_sparse/concatConcatV28sequential/layer1/embedding_lookup_sparse/Shape:output:07sequential/layer1/embedding_lookup_sparse/ones:output:0>sequential/layer1/embedding_lookup_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0sequential/layer1/embedding_lookup_sparse/concat�
1sequential/layer1/embedding_lookup_sparse/ReshapeReshapeIsequential/layer1/SparseFillEmptyRows/SparseFillEmptyRows:output_values:09sequential/layer1/embedding_lookup_sparse/concat:output:0*
T0*'
_output_shapes
:���������23
1sequential/layer1/embedding_lookup_sparse/Reshape�
-sequential/layer1/embedding_lookup_sparse/mulMul;sequential/layer1/embedding_lookup_sparse/GatherV2:output:0:sequential/layer1/embedding_lookup_sparse/Reshape:output:0*
T0*'
_output_shapes
:���������
2/
-sequential/layer1/embedding_lookup_sparse/mul�
)sequential/layer1/embedding_lookup_sparse
SegmentSum1sequential/layer1/embedding_lookup_sparse/mul:z:02sequential/layer1/embedding_lookup_sparse/Cast:y:0*
T0*
Tindices0*'
_output_shapes
:���������
2+
)sequential/layer1/embedding_lookup_sparse�
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02*
(sequential/layer1/BiasAdd/ReadVariableOp�
sequential/layer1/BiasAddBiasAdd2sequential/layer1/embedding_lookup_sparse:output:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential/layer1/BiasAdd�
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource*
_output_shapes

:
d*
dtype02)
'sequential/layer2/MatMul/ReadVariableOp�
sequential/layer2/MatMulMatMul"sequential/layer1/BiasAdd:output:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/layer2/MatMul�
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02*
(sequential/layer2/BiasAdd/ReadVariableOp�
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/layer2/BiasAdd�
sequential/layer2/ReluRelu"sequential/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential/layer2/Relu�
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource*
_output_shapes

:d<*
dtype02)
'sequential/layer3/MatMul/ReadVariableOp�
sequential/layer3/MatMulMatMul$sequential/layer2/Relu:activations:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
sequential/layer3/MatMul�
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype02*
(sequential/layer3/BiasAdd/ReadVariableOp�
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
sequential/layer3/BiasAdd�
sequential/layer3/ReluRelu"sequential/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:���������<2
sequential/layer3/Relu�
'sequential/layer4/MatMul/ReadVariableOpReadVariableOp0sequential_layer4_matmul_readvariableop_resource*
_output_shapes

:<*
dtype02)
'sequential/layer4/MatMul/ReadVariableOp�
sequential/layer4/MatMulMatMul$sequential/layer3/Relu:activations:0/sequential/layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/layer4/MatMul�
(sequential/layer4/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/layer4/BiasAdd/ReadVariableOp�
sequential/layer4/BiasAddBiasAdd"sequential/layer4/MatMul:product:00sequential/layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/layer4/BiasAdd�
sequential/layer4/ReluRelu"sequential/layer4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/layer4/Relu�
'sequential/layer5/MatMul/ReadVariableOpReadVariableOp0sequential_layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential/layer5/MatMul/ReadVariableOp�
sequential/layer5/MatMulMatMul$sequential/layer4/Relu:activations:0/sequential/layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/layer5/MatMul�
(sequential/layer5/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/layer5/BiasAdd/ReadVariableOp�
sequential/layer5/BiasAddBiasAdd"sequential/layer5/MatMul:product:00sequential/layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/layer5/BiasAdd�
sequential/layer5/SigmoidSigmoid"sequential/layer5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/layer5/Sigmoidw
sequential/layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A2
sequential/layer5/mul/y�
sequential/layer5/mulMulsequential/layer5/Sigmoid:y:0 sequential/layer5/mul/y:output:0*
T0*'
_output_shapes
:���������2
sequential/layer5/mulw
sequential/layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/layer5/add/y�
sequential/layer5/addAddV2sequential/layer5/mul:z:0 sequential/layer5/add/y:output:0*
T0*'
_output_shapes
:���������2
sequential/layer5/addt
IdentityIdentitysequential/layer5/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^sequential/layer1/BiasAdd/ReadVariableOp;^sequential/layer1/embedding_lookup_sparse/embedding_lookup)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp)^sequential/layer4/BiasAdd/ReadVariableOp(^sequential/layer4/MatMul/ReadVariableOp)^sequential/layer5/BiasAdd/ReadVariableOp(^sequential/layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 2T
(sequential/layer1/BiasAdd/ReadVariableOp(sequential/layer1/BiasAdd/ReadVariableOp2x
:sequential/layer1/embedding_lookup_sparse/embedding_lookup:sequential/layer1/embedding_lookup_sparse/embedding_lookup2T
(sequential/layer2/BiasAdd/ReadVariableOp(sequential/layer2/BiasAdd/ReadVariableOp2R
'sequential/layer2/MatMul/ReadVariableOp'sequential/layer2/MatMul/ReadVariableOp2T
(sequential/layer3/BiasAdd/ReadVariableOp(sequential/layer3/BiasAdd/ReadVariableOp2R
'sequential/layer3/MatMul/ReadVariableOp'sequential/layer3/MatMul/ReadVariableOp2T
(sequential/layer4/BiasAdd/ReadVariableOp(sequential/layer4/BiasAdd/ReadVariableOp2R
'sequential/layer4/MatMul/ReadVariableOp'sequential/layer4/MatMul/ReadVariableOp2T
(sequential/layer5/BiasAdd/ReadVariableOp(sequential/layer5/BiasAdd/ReadVariableOp2R
'sequential/layer5/MatMul/ReadVariableOp'sequential/layer5/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0:B>

_output_shapes
:
 
_user_specified_nameargs_0
�
+
__inference_loss_fn_0_18330
identity�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentity(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�

�
A__inference_layer2_layer_call_and_return_conditional_losses_18252

inputs0
matmul_readvariableop_resource:
d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_17973

args_0	
args_0_1
args_0_2	
unknown:
��

	unknown_0:

	unknown_1:
d
	unknown_2:d
	unknown_3:d<
	unknown_4:<
	unknown_5:<
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_175482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameargs_0:MI
#
_output_shapes
:���������
"
_user_specified_name
args_0_1:D@

_output_shapes
:
"
_user_specified_name
args_0_2
�
�
&__inference_layer2_layer_call_fn_18261

inputs
unknown:
d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_176222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
*__inference_sequential_layer_call_fn_18183

inputs	
inputs_1
inputs_2	
unknown:
��

	unknown_0:

	unknown_1:
d
	unknown_2:d
	unknown_3:d<
	unknown_4:<
	unknown_5:<
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_178232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������:���������:: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
�
�
A__inference_layer5_layer_call_and_return_conditional_losses_17677

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
SigmoidS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �A2
mul/y`
mulMulSigmoid:y:0mul/y:output:0*
T0*'
_output_shapes
:���������2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2mul:z:0add/y:output:0*
T0*'
_output_shapes
:���������2
addb
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
A__inference_layer1_layer_call_and_return_conditional_losses_17605

inputs	
inputs_1
inputs_2	B
.embedding_lookup_sparse_embedding_lookup_17579:
��
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�(embedding_lookup_sparse/embedding_lookup{
SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
SparseFillEmptyRows/Const�
'SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsinputsinputs_1inputs_2"SparseFillEmptyRows/Const:output:0*
T0*T
_output_shapesB
@:���������:���������:���������:���������2)
'SparseFillEmptyRows/SparseFillEmptyRows{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSlice8SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
+embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+embedding_lookup_sparse/strided_slice/stack�
-embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2/
-embedding_lookup_sparse/strided_slice/stack_1�
-embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-embedding_lookup_sparse/strided_slice/stack_2�
%embedding_lookup_sparse/strided_sliceStridedSlice8SparseFillEmptyRows/SparseFillEmptyRows:output_indices:04embedding_lookup_sparse/strided_slice/stack:output:06embedding_lookup_sparse/strided_slice/stack_1:output:06embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2'
%embedding_lookup_sparse/strided_slice�
embedding_lookup_sparse/UniqueUniquestrided_slice:output:0*
T0	*2
_output_shapes 
:���������:���������2 
embedding_lookup_sparse/Unique�
(embedding_lookup_sparse/embedding_lookupResourceGather.embedding_lookup_sparse_embedding_lookup_17579"embedding_lookup_sparse/Unique:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@embedding_lookup_sparse/embedding_lookup/17579*'
_output_shapes
:���������
*
dtype02*
(embedding_lookup_sparse/embedding_lookup�
1embedding_lookup_sparse/embedding_lookup/IdentityIdentity1embedding_lookup_sparse/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@embedding_lookup_sparse/embedding_lookup/17579*'
_output_shapes
:���������
23
1embedding_lookup_sparse/embedding_lookup/Identity�
3embedding_lookup_sparse/embedding_lookup/Identity_1Identity:embedding_lookup_sparse/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������
25
3embedding_lookup_sparse/embedding_lookup/Identity_1�
embedding_lookup_sparse/CastCast.embedding_lookup_sparse/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
embedding_lookup_sparse/Cast�
%embedding_lookup_sparse/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%embedding_lookup_sparse/GatherV2/axis�
 embedding_lookup_sparse/GatherV2GatherV2<embedding_lookup_sparse/embedding_lookup/Identity_1:output:0$embedding_lookup_sparse/Unique:idx:0.embedding_lookup_sparse/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:���������
2"
 embedding_lookup_sparse/GatherV2~
embedding_lookup_sparse/RankConst*
_output_shapes
: *
dtype0*
value	B :2
embedding_lookup_sparse/Rank�
embedding_lookup_sparse/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
embedding_lookup_sparse/sub/y�
embedding_lookup_sparse/subSub%embedding_lookup_sparse/Rank:output:0&embedding_lookup_sparse/sub/y:output:0*
T0*
_output_shapes
: 2
embedding_lookup_sparse/sub�
&embedding_lookup_sparse/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&embedding_lookup_sparse/ExpandDims/dim�
"embedding_lookup_sparse/ExpandDims
ExpandDimsembedding_lookup_sparse/sub:z:0/embedding_lookup_sparse/ExpandDims/dim:output:0*
T0*
_output_shapes
:2$
"embedding_lookup_sparse/ExpandDims�
"embedding_lookup_sparse/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"embedding_lookup_sparse/ones/Const�
embedding_lookup_sparse/onesFill+embedding_lookup_sparse/ExpandDims:output:0+embedding_lookup_sparse/ones/Const:output:0*
T0*
_output_shapes
:2
embedding_lookup_sparse/ones�
embedding_lookup_sparse/ShapeShape7SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0*
_output_shapes
:2
embedding_lookup_sparse/Shape�
#embedding_lookup_sparse/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#embedding_lookup_sparse/concat/axis�
embedding_lookup_sparse/concatConcatV2&embedding_lookup_sparse/Shape:output:0%embedding_lookup_sparse/ones:output:0,embedding_lookup_sparse/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
embedding_lookup_sparse/concat�
embedding_lookup_sparse/ReshapeReshape7SparseFillEmptyRows/SparseFillEmptyRows:output_values:0'embedding_lookup_sparse/concat:output:0*
T0*'
_output_shapes
:���������2!
embedding_lookup_sparse/Reshape�
embedding_lookup_sparse/mulMul)embedding_lookup_sparse/GatherV2:output:0(embedding_lookup_sparse/Reshape:output:0*
T0*'
_output_shapes
:���������
2
embedding_lookup_sparse/mul�
embedding_lookup_sparse
SegmentSumembedding_lookup_sparse/mul:z:0 embedding_lookup_sparse/Cast:y:0*
T0*
Tindices0*'
_output_shapes
:���������
2
embedding_lookup_sparse�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd embedding_lookup_sparse:output:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdd�
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp)^embedding_lookup_sparse/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2T
(embedding_lookup_sparse/embedding_lookup(embedding_lookup_sparse/embedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_18611
file_prefix2
assignvariableop_layer1_kernel:
��
,
assignvariableop_1_layer1_bias:
2
 assignvariableop_2_layer2_kernel:
d,
assignvariableop_3_layer2_bias:d2
 assignvariableop_4_layer3_kernel:d<,
assignvariableop_5_layer3_bias:<2
 assignvariableop_6_layer4_kernel:<,
assignvariableop_7_layer4_bias:2
 assignvariableop_8_layer5_kernel:,
assignvariableop_9_layer5_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1:  
assignvariableop_19_nc: 
assignvariableop_20_t: <
(assignvariableop_21_adam_layer1_kernel_m:
��
4
&assignvariableop_22_adam_layer1_bias_m:
:
(assignvariableop_23_adam_layer2_kernel_m:
d4
&assignvariableop_24_adam_layer2_bias_m:d:
(assignvariableop_25_adam_layer3_kernel_m:d<4
&assignvariableop_26_adam_layer3_bias_m:<:
(assignvariableop_27_adam_layer4_kernel_m:<4
&assignvariableop_28_adam_layer4_bias_m::
(assignvariableop_29_adam_layer5_kernel_m:4
&assignvariableop_30_adam_layer5_bias_m:<
(assignvariableop_31_adam_layer1_kernel_v:
��
4
&assignvariableop_32_adam_layer1_bias_v:
:
(assignvariableop_33_adam_layer2_kernel_v:
d4
&assignvariableop_34_adam_layer2_bias_v:d:
(assignvariableop_35_adam_layer3_kernel_v:d<4
&assignvariableop_36_adam_layer3_bias_v:<:
(assignvariableop_37_adam_layer4_kernel_v:<4
&assignvariableop_38_adam_layer4_bias_v::
(assignvariableop_39_adam_layer5_kernel_v:4
&assignvariableop_40_adam_layer5_bias_v:
identity_42��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/2/nc/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/2/t/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_ncIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_tIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_layer1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_layer1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_layer1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_layer1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_layer2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_layer2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer5_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer5_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41f
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_42�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
A__inference_layer3_layer_call_and_return_conditional_losses_18272

inputs0
matmul_readvariableop_resource:d<-
biasadd_readvariableop_resource:<
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d<*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������<2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������<2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������<2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
args_0/
serving_default_args_0:0	���������
9
args_0_1-
serving_default_args_0_1:0���������
0
args_0_2$
serving_default_args_0_2:0	:
layer50
StatefulPartitionedCall:0���������tensorflow/serving/predict:�j
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
*q&call_and_return_all_conditional_losses
r__call__
s_default_save_signature"
_tf_keras_sequential
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*v&call_and_return_all_conditional_losses
w__call__"
_tf_keras_layer
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"
_tf_keras_layer
�

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*z&call_and_return_all_conditional_losses
{__call__"
_tf_keras_layer
�

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
*|&call_and_return_all_conditional_losses
}__call__"
_tf_keras_layer
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem]m^m_m`mambmcmd$me%mfvgvhvivjvkvlvmvn$vo%vp"
	optimizer
'
~0"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
�
/non_trainable_variables
0layer_regularization_losses
regularization_losses
trainable_variables
1layer_metrics

2layers
		variables
3metrics
r__call__
s_default_save_signature
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
!:
��
2layer1/kernel
:
2layer1/bias
.
0
1"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
4non_trainable_variables
	variables
regularization_losses
trainable_variables
5layer_metrics

6layers
7layer_regularization_losses
8metrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
:
d2layer2/kernel
:d2layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
9non_trainable_variables
	variables
regularization_losses
trainable_variables
:layer_metrics

;layers
<layer_regularization_losses
=metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
:d<2layer3/kernel
:<2layer3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
>non_trainable_variables
	variables
regularization_losses
trainable_variables
?layer_metrics

@layers
Alayer_regularization_losses
Bmetrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
:<2layer4/kernel
:2layer4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Cnon_trainable_variables
 	variables
!regularization_losses
"trainable_variables
Dlayer_metrics

Elayers
Flayer_regularization_losses
Gmetrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
:2layer5/kernel
:2layer5/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
Hnon_trainable_variables
&	variables
'regularization_losses
(trainable_variables
Ilayer_metrics

Jlayers
Klayer_regularization_losses
Lmetrics
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metric
^
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"
_tf_keras_metric
a
Ync
Y	n_correct
Zt
	Ztotal
[	variables
\	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
:  (2nc
:  (2t
.
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
&:$
��
2Adam/layer1/kernel/m
:
2Adam/layer1/bias/m
$:"
d2Adam/layer2/kernel/m
:d2Adam/layer2/bias/m
$:"d<2Adam/layer3/kernel/m
:<2Adam/layer3/bias/m
$:"<2Adam/layer4/kernel/m
:2Adam/layer4/bias/m
$:"2Adam/layer5/kernel/m
:2Adam/layer5/bias/m
&:$
��
2Adam/layer1/kernel/v
:
2Adam/layer1/bias/v
$:"
d2Adam/layer2/kernel/v
:d2Adam/layer2/bias/v
$:"d<2Adam/layer3/kernel/v
:<2Adam/layer3/bias/v
$:"<2Adam/layer4/kernel/v
:2Adam/layer4/bias/v
$:"2Adam/layer5/kernel/v
:2Adam/layer5/bias/v
�2�
E__inference_sequential_layer_call_and_return_conditional_losses_18051
E__inference_sequential_layer_call_and_return_conditional_losses_18129�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_sequential_layer_call_fn_18156
*__inference_sequential_layer_call_fn_18183�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_17548args_0args_0_1args_0_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_layer1_layer_call_and_return_conditional_losses_18230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_layer1_layer_call_fn_18241�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_layer2_layer_call_and_return_conditional_losses_18252�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_layer2_layer_call_fn_18261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_layer3_layer_call_and_return_conditional_losses_18272�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_layer3_layer_call_fn_18281�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_layer4_layer_call_and_return_conditional_losses_18292�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_layer4_layer_call_fn_18301�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_layer5_layer_call_and_return_conditional_losses_18316�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_layer5_layer_call_fn_18325�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_18330�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
#__inference_signature_wrapper_17973args_0args_0_1args_0_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_17548�
$%O�L
E�B
@�='�$
�������������������
�SparseTensorSpec
� "/�,
*
layer5 �
layer5����������
A__inference_layer1_layer_call_and_return_conditional_losses_18230|O�L
E�B
@�='�$
�������������������
�SparseTensorSpec
� "%�"
�
0���������

� �
&__inference_layer1_layer_call_fn_18241oO�L
E�B
@�='�$
�������������������
�SparseTensorSpec
� "����������
�
A__inference_layer2_layer_call_and_return_conditional_losses_18252\/�,
%�"
 �
inputs���������

� "%�"
�
0���������d
� y
&__inference_layer2_layer_call_fn_18261O/�,
%�"
 �
inputs���������

� "����������d�
A__inference_layer3_layer_call_and_return_conditional_losses_18272\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������<
� y
&__inference_layer3_layer_call_fn_18281O/�,
%�"
 �
inputs���������d
� "����������<�
A__inference_layer4_layer_call_and_return_conditional_losses_18292\/�,
%�"
 �
inputs���������<
� "%�"
�
0���������
� y
&__inference_layer4_layer_call_fn_18301O/�,
%�"
 �
inputs���������<
� "�����������
A__inference_layer5_layer_call_and_return_conditional_losses_18316\$%/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
&__inference_layer5_layer_call_fn_18325O$%/�,
%�"
 �
inputs���������
� "����������7
__inference_loss_fn_0_18330�

� 
� "� �
E__inference_sequential_layer_call_and_return_conditional_losses_18051�
$%W�T
M�J
@�='�$
�������������������
�SparseTensorSpec
p 

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_18129�
$%W�T
M�J
@�='�$
�������������������
�SparseTensorSpec
p

 
� "%�"
�
0���������
� �
*__inference_sequential_layer_call_fn_18156
$%W�T
M�J
@�='�$
�������������������
�SparseTensorSpec
p 

 
� "�����������
*__inference_sequential_layer_call_fn_18183
$%W�T
M�J
@�='�$
�������������������
�SparseTensorSpec
p

 
� "�����������
#__inference_signature_wrapper_17973�
$%���
� 
~�{
*
args_0 �
args_0���������	
*
args_0_1�
args_0_1���������
!
args_0_2�
args_0_2	"/�,
*
layer5 �
layer5���������