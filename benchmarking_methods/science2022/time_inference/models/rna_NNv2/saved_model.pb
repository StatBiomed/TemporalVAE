СШ
х╣
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.12v2.6.0-101-g3aa40c3ce9d8вЧ
w
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	│*
shared_namelayer1/kernel
p
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes
:	│*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:*
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

:d*
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
:d2*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:d2*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:2*
dtype0
v
layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namelayer4/kernel
o
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*
_output_shapes

:2*
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
Е
Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	│*%
shared_nameAdam/layer1/kernel/m
~
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*
_output_shapes
:	│*
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
:*
dtype0
Д
Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

:d*
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
Д
Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*%
shared_nameAdam/layer3/kernel/m
}
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*
_output_shapes

:d2*
dtype0
|
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*#
shared_nameAdam/layer3/bias/m
u
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes
:2*
dtype0
Д
Adam/layer4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/layer4/kernel/m
}
(Adam/layer4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/m*
_output_shapes

:2*
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
Д
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
Е
Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	│*%
shared_nameAdam/layer1/kernel/v
~
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*
_output_shapes
:	│*
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
:*
dtype0
Д
Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

:d*
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
Д
Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*%
shared_nameAdam/layer3/kernel/v
}
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*
_output_shapes

:d2*
dtype0
|
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*#
shared_nameAdam/layer3/bias/v
u
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes
:2*
dtype0
Д
Adam/layer4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/layer4/kernel/v
}
(Adam/layer4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer4/kernel/v*
_output_shapes

:2*
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
Д
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
й8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ф7
value┌7B╫7 B╨7
┤
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
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
Ї
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
н
regularization_losses
trainable_variables

/layers
0non_trainable_variables
1layer_regularization_losses
2layer_metrics
		variables
3metrics
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
trainable_variables

4layers
5non_trainable_variables
	variables
6layer_regularization_losses
7layer_metrics
8metrics
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
trainable_variables

9layers
:non_trainable_variables
	variables
;layer_regularization_losses
<layer_metrics
=metrics
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
trainable_variables

>layers
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
Bmetrics
YW
VARIABLE_VALUElayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
 regularization_losses
!trainable_variables

Clayers
Dnon_trainable_variables
"	variables
Elayer_regularization_losses
Flayer_metrics
Gmetrics
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
н
&regularization_losses
'trainable_variables

Hlayers
Inon_trainable_variables
(	variables
Jlayer_regularization_losses
Klayer_metrics
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
#
0
1
2
3
4
 
 
 
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
Б
serving_default_layer1_inputPlaceholder*(
_output_shapes
:         │*
dtype0*
shape:         │
╓
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer1_inputlayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_89593
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
У
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
GPU 2J 8В *'
f"R 
__inference__traced_save_89991
┌
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_90124п∙
С5
╦
E__inference_sequential_layer_call_and_return_conditional_losses_89688

inputs8
%layer1_matmul_readvariableop_resource:	│4
&layer1_biasadd_readvariableop_resource:7
%layer2_matmul_readvariableop_resource:d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:d24
&layer3_biasadd_readvariableop_resource:27
%layer4_matmul_readvariableop_resource:24
&layer4_biasadd_readvariableop_resource:7
%layer5_matmul_readvariableop_resource:4
&layer5_biasadd_readvariableop_resource:
identityИвlayer1/BiasAdd/ReadVariableOpвlayer1/MatMul/ReadVariableOpвlayer2/BiasAdd/ReadVariableOpвlayer2/MatMul/ReadVariableOpвlayer3/BiasAdd/ReadVariableOpвlayer3/MatMul/ReadVariableOpвlayer4/BiasAdd/ReadVariableOpвlayer4/MatMul/ReadVariableOpвlayer5/BiasAdd/ReadVariableOpвlayer5/MatMul/ReadVariableOpг
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	│*
dtype02
layer1/MatMul/ReadVariableOpИ
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer1/MatMulб
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOpЭ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer1/BiasAddв
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
layer2/MatMul/ReadVariableOpЩ
layer2/MatMulMatMullayer1/BiasAdd:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
layer2/Reluв
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
layer3/MatMul/ReadVariableOpЫ
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer3/MatMulб
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
layer3/BiasAdd/ReadVariableOpЭ
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer3/Reluв
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer4/MatMul/ReadVariableOpЫ
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer4/MatMulб
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer4/BiasAdd/ReadVariableOpЭ
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
layer4/Reluв
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer5/MatMul/ReadVariableOpЫ
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer5/MatMulб
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer5/BiasAdd/ReadVariableOpЭ
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer5/BiasAddm
layer5/TanhTanhlayer5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
layer5/Tanha
layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
layer5/add/y{

layer5/addAddV2layer5/Tanh:y:0layer5/add/y:output:0*
T0*'
_output_shapes
:         2

layer5/adda
layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
layer5/mul/yx

layer5/mulMullayer5/add:z:0layer5/mul/y:output:0*
T0*'
_output_shapes
:         2

layer5/mule
layer5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer5/add_1/yА
layer5/add_1AddV2layer5/mul:z:0layer5/add_1/y:output:0*
T0*'
_output_shapes
:         2
layer5/add_1З
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentitylayer5/add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

IdentityЙ
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
ы
У
&__inference_layer5_layer_call_fn_89823

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_893132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 

Є
A__inference_layer3_layer_call_and_return_conditional_losses_89794

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
 

Є
A__inference_layer2_layer_call_and_return_conditional_losses_89774

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ю
Є
A__inference_layer5_layer_call_and_return_conditional_losses_89840

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
add/y_
addAddV2Tanh:y:0add/y:output:0*
T0*'
_output_shapes
:         2
addS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/y\
mulMuladd:z:0mul/y:output:0*
T0*'
_output_shapes
:         2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:         2
add_1d
IdentityIdentity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 

Є
A__inference_layer4_layer_call_and_return_conditional_losses_89290

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
и
╝
E__inference_sequential_layer_call_and_return_conditional_losses_89559
layer1_input
layer1_89532:	│
layer1_89534:
layer2_89537:d
layer2_89539:d
layer3_89542:d2
layer3_89544:2
layer4_89547:2
layer4_89549:
layer5_89552:
layer5_89554:
identityИвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallвlayer3/StatefulPartitionedCallвlayer4/StatefulPartitionedCallвlayer5/StatefulPartitionedCallН
layer1/StatefulPartitionedCallStatefulPartitionedCalllayer1_inputlayer1_89532layer1_89534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_892392 
layer1/StatefulPartitionedCallи
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_89537layer2_89539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_892562 
layer2/StatefulPartitionedCallи
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_89542layer3_89544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_892732 
layer3/StatefulPartitionedCallи
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_89547layer4_89549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_892902 
layer4/StatefulPartitionedCallи
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_89552layer5_89554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_893132 
layer5/StatefulPartitionedCallЗ
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/ConstВ
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityє
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
ы
У
&__inference_layer2_layer_call_fn_89763

inputs
unknown:d
	unknown_0:d
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_892562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 

Є
A__inference_layer2_layer_call_and_return_conditional_losses_89256

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
по
╫
!__inference__traced_restore_90124
file_prefix1
assignvariableop_layer1_kernel:	│,
assignvariableop_1_layer1_bias:2
 assignvariableop_2_layer2_kernel:d,
assignvariableop_3_layer2_bias:d2
 assignvariableop_4_layer3_kernel:d2,
assignvariableop_5_layer3_bias:22
 assignvariableop_6_layer4_kernel:2,
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
assignvariableop_20_t: ;
(assignvariableop_21_adam_layer1_kernel_m:	│4
&assignvariableop_22_adam_layer1_bias_m::
(assignvariableop_23_adam_layer2_kernel_m:d4
&assignvariableop_24_adam_layer2_bias_m:d:
(assignvariableop_25_adam_layer3_kernel_m:d24
&assignvariableop_26_adam_layer3_bias_m:2:
(assignvariableop_27_adam_layer4_kernel_m:24
&assignvariableop_28_adam_layer4_bias_m::
(assignvariableop_29_adam_layer5_kernel_m:4
&assignvariableop_30_adam_layer5_bias_m:;
(assignvariableop_31_adam_layer1_kernel_v:	│4
&assignvariableop_32_adam_layer1_bias_v::
(assignvariableop_33_adam_layer2_kernel_v:d4
&assignvariableop_34_adam_layer2_bias_v:d:
(assignvariableop_35_adam_layer3_kernel_v:d24
&assignvariableop_36_adam_layer3_bias_v:2:
(assignvariableop_37_adam_layer4_kernel_v:24
&assignvariableop_38_adam_layer4_bias_v::
(assignvariableop_39_adam_layer5_kernel_v:4
&assignvariableop_40_adam_layer5_bias_v:
identity_42ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*ў
valueэBъ*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/2/nc/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/2/t/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesт
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesА
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╛
_output_shapesл
и::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1г
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8е
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10е
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11з
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ж
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14о
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15б
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16б
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17г
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18г
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ю
AssignVariableOp_19AssignVariableOpassignvariableop_19_ncIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Э
AssignVariableOp_20AssignVariableOpassignvariableop_20_tIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21░
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_layer1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22о
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_layer1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23░
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24о
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25░
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26о
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27░
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28о
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29░
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer5_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30о
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer5_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_layer1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32о
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_layer1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33░
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_layer2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34о
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_layer2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35░
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36о
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37░
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38о
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39░
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer5_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40о
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer5_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpф
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41f
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_42╠
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
Ц
╢
E__inference_sequential_layer_call_and_return_conditional_losses_89451

inputs
layer1_89424:	│
layer1_89426:
layer2_89429:d
layer2_89431:d
layer3_89434:d2
layer3_89436:2
layer4_89439:2
layer4_89441:
layer5_89444:
layer5_89446:
identityИвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallвlayer3/StatefulPartitionedCallвlayer4/StatefulPartitionedCallвlayer5/StatefulPartitionedCallЗ
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_89424layer1_89426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_892392 
layer1/StatefulPartitionedCallи
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_89429layer2_89431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_892562 
layer2/StatefulPartitionedCallи
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_89434layer3_89436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_892732 
layer3/StatefulPartitionedCallи
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_89439layer4_89441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_892902 
layer4/StatefulPartitionedCallи
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_89444layer5_89446*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_893132 
layer5/StatefulPartitionedCallЗ
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/ConstВ
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityє
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
╪

Ў
*__inference_sequential_layer_call_fn_89344
layer1_input
unknown:	│
	unknown_0:
	unknown_1:d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
	unknown_7:
	unknown_8:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_893212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
╪

Ў
*__inference_sequential_layer_call_fn_89499
layer1_input
unknown:	│
	unknown_0:
	unknown_1:d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
	unknown_7:
	unknown_8:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_894512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
м

я
#__inference_signature_wrapper_89593
layer1_input
unknown:	│
	unknown_0:
	unknown_1:d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
	unknown_7:
	unknown_8:
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCalllayer1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_892212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
ы
У
&__inference_layer3_layer_call_fn_89783

inputs
unknown:d2
	unknown_0:2
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_892732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ь@
И	
 __inference__wrapped_model_89221
layer1_inputC
0sequential_layer1_matmul_readvariableop_resource:	│?
1sequential_layer1_biasadd_readvariableop_resource:B
0sequential_layer2_matmul_readvariableop_resource:d?
1sequential_layer2_biasadd_readvariableop_resource:dB
0sequential_layer3_matmul_readvariableop_resource:d2?
1sequential_layer3_biasadd_readvariableop_resource:2B
0sequential_layer4_matmul_readvariableop_resource:2?
1sequential_layer4_biasadd_readvariableop_resource:B
0sequential_layer5_matmul_readvariableop_resource:?
1sequential_layer5_biasadd_readvariableop_resource:
identityИв(sequential/layer1/BiasAdd/ReadVariableOpв'sequential/layer1/MatMul/ReadVariableOpв(sequential/layer2/BiasAdd/ReadVariableOpв'sequential/layer2/MatMul/ReadVariableOpв(sequential/layer3/BiasAdd/ReadVariableOpв'sequential/layer3/MatMul/ReadVariableOpв(sequential/layer4/BiasAdd/ReadVariableOpв'sequential/layer4/MatMul/ReadVariableOpв(sequential/layer5/BiasAdd/ReadVariableOpв'sequential/layer5/MatMul/ReadVariableOp─
'sequential/layer1/MatMul/ReadVariableOpReadVariableOp0sequential_layer1_matmul_readvariableop_resource*
_output_shapes
:	│*
dtype02)
'sequential/layer1/MatMul/ReadVariableOpп
sequential/layer1/MatMulMatMullayer1_input/sequential/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer1/MatMul┬
(sequential/layer1/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/layer1/BiasAdd/ReadVariableOp╔
sequential/layer1/BiasAddBiasAdd"sequential/layer1/MatMul:product:00sequential/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer1/BiasAdd├
'sequential/layer2/MatMul/ReadVariableOpReadVariableOp0sequential_layer2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02)
'sequential/layer2/MatMul/ReadVariableOp┼
sequential/layer2/MatMulMatMul"sequential/layer1/BiasAdd:output:0/sequential/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential/layer2/MatMul┬
(sequential/layer2/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02*
(sequential/layer2/BiasAdd/ReadVariableOp╔
sequential/layer2/BiasAddBiasAdd"sequential/layer2/MatMul:product:00sequential/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential/layer2/BiasAddО
sequential/layer2/ReluRelu"sequential/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
sequential/layer2/Relu├
'sequential/layer3/MatMul/ReadVariableOpReadVariableOp0sequential_layer3_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02)
'sequential/layer3/MatMul/ReadVariableOp╟
sequential/layer3/MatMulMatMul$sequential/layer2/Relu:activations:0/sequential/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer3/MatMul┬
(sequential/layer3/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02*
(sequential/layer3/BiasAdd/ReadVariableOp╔
sequential/layer3/BiasAddBiasAdd"sequential/layer3/MatMul:product:00sequential/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
sequential/layer3/BiasAddО
sequential/layer3/ReluRelu"sequential/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:         22
sequential/layer3/Relu├
'sequential/layer4/MatMul/ReadVariableOpReadVariableOp0sequential_layer4_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02)
'sequential/layer4/MatMul/ReadVariableOp╟
sequential/layer4/MatMulMatMul$sequential/layer3/Relu:activations:0/sequential/layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer4/MatMul┬
(sequential/layer4/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/layer4/BiasAdd/ReadVariableOp╔
sequential/layer4/BiasAddBiasAdd"sequential/layer4/MatMul:product:00sequential/layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer4/BiasAddО
sequential/layer4/ReluRelu"sequential/layer4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/layer4/Relu├
'sequential/layer5/MatMul/ReadVariableOpReadVariableOp0sequential_layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential/layer5/MatMul/ReadVariableOp╟
sequential/layer5/MatMulMatMul$sequential/layer4/Relu:activations:0/sequential/layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer5/MatMul┬
(sequential/layer5/BiasAdd/ReadVariableOpReadVariableOp1sequential_layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/layer5/BiasAdd/ReadVariableOp╔
sequential/layer5/BiasAddBiasAdd"sequential/layer5/MatMul:product:00sequential/layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/layer5/BiasAddО
sequential/layer5/TanhTanh"sequential/layer5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/layer5/Tanhw
sequential/layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sequential/layer5/add/yз
sequential/layer5/addAddV2sequential/layer5/Tanh:y:0 sequential/layer5/add/y:output:0*
T0*'
_output_shapes
:         2
sequential/layer5/addw
sequential/layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
sequential/layer5/mul/yд
sequential/layer5/mulMulsequential/layer5/add:z:0 sequential/layer5/mul/y:output:0*
T0*'
_output_shapes
:         2
sequential/layer5/mul{
sequential/layer5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/layer5/add_1/yм
sequential/layer5/add_1AddV2sequential/layer5/mul:z:0"sequential/layer5/add_1/y:output:0*
T0*'
_output_shapes
:         2
sequential/layer5/add_1v
IdentityIdentitysequential/layer5/add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identityў
NoOpNoOp)^sequential/layer1/BiasAdd/ReadVariableOp(^sequential/layer1/MatMul/ReadVariableOp)^sequential/layer2/BiasAdd/ReadVariableOp(^sequential/layer2/MatMul/ReadVariableOp)^sequential/layer3/BiasAdd/ReadVariableOp(^sequential/layer3/MatMul/ReadVariableOp)^sequential/layer4/BiasAdd/ReadVariableOp(^sequential/layer4/MatMul/ReadVariableOp)^sequential/layer5/BiasAdd/ReadVariableOp(^sequential/layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2T
(sequential/layer1/BiasAdd/ReadVariableOp(sequential/layer1/BiasAdd/ReadVariableOp2R
'sequential/layer1/MatMul/ReadVariableOp'sequential/layer1/MatMul/ReadVariableOp2T
(sequential/layer2/BiasAdd/ReadVariableOp(sequential/layer2/BiasAdd/ReadVariableOp2R
'sequential/layer2/MatMul/ReadVariableOp'sequential/layer2/MatMul/ReadVariableOp2T
(sequential/layer3/BiasAdd/ReadVariableOp(sequential/layer3/BiasAdd/ReadVariableOp2R
'sequential/layer3/MatMul/ReadVariableOp'sequential/layer3/MatMul/ReadVariableOp2T
(sequential/layer4/BiasAdd/ReadVariableOp(sequential/layer4/BiasAdd/ReadVariableOp2R
'sequential/layer4/MatMul/ReadVariableOp'sequential/layer4/MatMul/ReadVariableOp2T
(sequential/layer5/BiasAdd/ReadVariableOp(sequential/layer5/BiasAdd/ReadVariableOp2R
'sequential/layer5/MatMul/ReadVariableOp'sequential/layer5/MatMul/ReadVariableOp:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
Ц
╢
E__inference_sequential_layer_call_and_return_conditional_losses_89321

inputs
layer1_89240:	│
layer1_89242:
layer2_89257:d
layer2_89259:d
layer3_89274:d2
layer3_89276:2
layer4_89291:2
layer4_89293:
layer5_89314:
layer5_89316:
identityИвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallвlayer3/StatefulPartitionedCallвlayer4/StatefulPartitionedCallвlayer5/StatefulPartitionedCallЗ
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_89240layer1_89242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_892392 
layer1/StatefulPartitionedCallи
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_89257layer2_89259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_892562 
layer2/StatefulPartitionedCallи
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_89274layer3_89276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_892732 
layer3/StatefulPartitionedCallи
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_89291layer4_89293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_892902 
layer4/StatefulPartitionedCallи
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_89314layer5_89316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_893132 
layer5/StatefulPartitionedCallЗ
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/ConstВ
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityє
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
и
╝
E__inference_sequential_layer_call_and_return_conditional_losses_89529
layer1_input
layer1_89502:	│
layer1_89504:
layer2_89507:d
layer2_89509:d
layer3_89512:d2
layer3_89514:2
layer4_89517:2
layer4_89519:
layer5_89522:
layer5_89524:
identityИвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallвlayer3/StatefulPartitionedCallвlayer4/StatefulPartitionedCallвlayer5/StatefulPartitionedCallН
layer1/StatefulPartitionedCallStatefulPartitionedCalllayer1_inputlayer1_89502layer1_89504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_892392 
layer1/StatefulPartitionedCallи
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_89507layer2_89509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer2_layer_call_and_return_conditional_losses_892562 
layer2/StatefulPartitionedCallи
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_89512layer3_89514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer3_layer_call_and_return_conditional_losses_892732 
layer3/StatefulPartitionedCallи
layer4/StatefulPartitionedCallStatefulPartitionedCall'layer3/StatefulPartitionedCall:output:0layer4_89517layer4_89519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_892902 
layer4/StatefulPartitionedCallи
layer5/StatefulPartitionedCallStatefulPartitionedCall'layer4/StatefulPartitionedCall:output:0layer5_89522layer5_89524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer5_layer_call_and_return_conditional_losses_893132 
layer5/StatefulPartitionedCallЗ
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/ConstВ
IdentityIdentity'layer5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityє
NoOpNoOp^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall:V R
(
_output_shapes
:         │
&
_user_specified_namelayer1_input
 

Є
A__inference_layer3_layer_call_and_return_conditional_losses_89273

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
▒
є
A__inference_layer1_layer_call_and_return_conditional_losses_89239

inputs1
matmul_readvariableop_resource:	│-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	│*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЗ
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
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
Ю
Є
A__inference_layer5_layer_call_and_return_conditional_losses_89313

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
add/y_
addAddV2Tanh:y:0add/y:output:0*
T0*'
_output_shapes
:         2
addS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
mul/y\
mulMuladd:z:0mul/y:output:0*
T0*'
_output_shapes
:         2
mulW
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
add_1/yd
add_1AddV2mul:z:0add_1/y:output:0*
T0*'
_output_shapes
:         2
add_1d
IdentityIdentity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
У
&__inference_layer4_layer_call_fn_89803

inputs
unknown:2
	unknown_0:
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer4_layer_call_and_return_conditional_losses_892902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
ю
Ф
&__inference_layer1_layer_call_fn_89743

inputs
unknown:	│
	unknown_0:
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_layer1_layer_call_and_return_conditional_losses_892392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
БT
■
__inference__traced_save_89991
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

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameх
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*ў
valueэBъ*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/2/nc/.ATTRIBUTES/VARIABLE_VALUEB0keras_api/metrics/2/t/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names▄
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╥
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_nc_read_readvariableopsavev2_t_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop/savev2_adam_layer4_kernel_m_read_readvariableop-savev2_adam_layer4_bias_m_read_readvariableop/savev2_adam_layer5_kernel_m_read_readvariableop-savev2_adam_layer5_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop/savev2_adam_layer4_kernel_v_read_readvariableop-savev2_adam_layer4_bias_v_read_readvariableop/savev2_adam_layer5_kernel_v_read_readvariableop-savev2_adam_layer5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*в
_input_shapesР
Н: :	│::d:d:d2:2:2:::: : : : : : : : : : : :	│::d:d:d2:2:2::::	│::d:d:d2:2:2:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	│: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:2: 
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
: :%!

_output_shapes
:	│: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::% !

_output_shapes
:	│: !

_output_shapes
::$" 

_output_shapes

:d: #

_output_shapes
:d:$$ 

_output_shapes

:d2: %

_output_shapes
:2:$& 

_output_shapes

:2: '
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
С5
╦
E__inference_sequential_layer_call_and_return_conditional_losses_89733

inputs8
%layer1_matmul_readvariableop_resource:	│4
&layer1_biasadd_readvariableop_resource:7
%layer2_matmul_readvariableop_resource:d4
&layer2_biasadd_readvariableop_resource:d7
%layer3_matmul_readvariableop_resource:d24
&layer3_biasadd_readvariableop_resource:27
%layer4_matmul_readvariableop_resource:24
&layer4_biasadd_readvariableop_resource:7
%layer5_matmul_readvariableop_resource:4
&layer5_biasadd_readvariableop_resource:
identityИвlayer1/BiasAdd/ReadVariableOpвlayer1/MatMul/ReadVariableOpвlayer2/BiasAdd/ReadVariableOpвlayer2/MatMul/ReadVariableOpвlayer3/BiasAdd/ReadVariableOpвlayer3/MatMul/ReadVariableOpвlayer4/BiasAdd/ReadVariableOpвlayer4/MatMul/ReadVariableOpвlayer5/BiasAdd/ReadVariableOpвlayer5/MatMul/ReadVariableOpг
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes
:	│*
dtype02
layer1/MatMul/ReadVariableOpИ
layer1/MatMulMatMulinputs$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer1/MatMulб
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOpЭ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer1/BiasAddв
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
layer2/MatMul/ReadVariableOpЩ
layer2/MatMulMatMullayer1/BiasAdd:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
layer2/BiasAddm
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
layer2/Reluв
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02
layer3/MatMul/ReadVariableOpЫ
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer3/MatMulб
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
layer3/BiasAdd/ReadVariableOpЭ
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
layer3/BiasAddm
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:         22
layer3/Reluв
layer4/MatMul/ReadVariableOpReadVariableOp%layer4_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
layer4/MatMul/ReadVariableOpЫ
layer4/MatMulMatMullayer3/Relu:activations:0$layer4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer4/MatMulб
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer4/BiasAdd/ReadVariableOpЭ
layer4/BiasAddBiasAddlayer4/MatMul:product:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer4/BiasAddm
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*'
_output_shapes
:         2
layer4/Reluв
layer5/MatMul/ReadVariableOpReadVariableOp%layer5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
layer5/MatMul/ReadVariableOpЫ
layer5/MatMulMatMullayer4/Relu:activations:0$layer5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer5/MatMulб
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer5/BiasAdd/ReadVariableOpЭ
layer5/BiasAddBiasAddlayer5/MatMul:product:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer5/BiasAddm
layer5/TanhTanhlayer5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
layer5/Tanha
layer5/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
layer5/add/y{

layer5/addAddV2layer5/Tanh:y:0layer5/add/y:output:0*
T0*'
_output_shapes
:         2

layer5/adda
layer5/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
layer5/mul/yx

layer5/mulMullayer5/add:z:0layer5/mul/y:output:0*
T0*'
_output_shapes
:         2

layer5/mule
layer5/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
layer5/add_1/yА
layer5/add_1AddV2layer5/mul:z:0layer5/add_1/y:output:0*
T0*'
_output_shapes
:         2
layer5/add_1З
layer1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
layer1/kernel/Regularizer/Constk
IdentityIdentitylayer5/add_1:z:0^NoOp*
T0*'
_output_shapes
:         2

IdentityЙ
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/MatMul/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/MatMul/ReadVariableOplayer4/MatMul/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/MatMul/ReadVariableOplayer5/MatMul/ReadVariableOp:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
▒
є
A__inference_layer1_layer_call_and_return_conditional_losses_89754

inputs1
matmul_readvariableop_resource:	│-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	│*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddЗ
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
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         │: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
 

Є
A__inference_layer4_layer_call_and_return_conditional_losses_89814

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
╞

Ё
*__inference_sequential_layer_call_fn_89643

inputs
unknown:	│
	unknown_0:
	unknown_1:d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
	unknown_7:
	unknown_8:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_894512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
╞

Ё
*__inference_sequential_layer_call_fn_89618

inputs
unknown:	│
	unknown_0:
	unknown_1:d
	unknown_2:d
	unknown_3:d2
	unknown_4:2
	unknown_5:2
	unknown_6:
	unknown_7:
	unknown_8:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_893212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         │: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         │
 
_user_specified_nameinputs
А
+
__inference_loss_fn_0_89845
identityЗ
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
_input_shapes "иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┤
serving_defaultа
F
layer1_input6
serving_default_layer1_input:0         │:
layer50
StatefulPartitionedCall:0         tensorflow/serving/predict:Ъo
й
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
q__call__
r_default_save_signature
*s&call_and_return_all_conditional_losses"
_tf_keras_sequential
╗

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
З
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
╩
regularization_losses
trainable_variables

/layers
0non_trainable_variables
1layer_regularization_losses
2layer_metrics
		variables
3metrics
q__call__
r_default_save_signature
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
 :	│2layer1/kernel
:2layer1/bias
'
~0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
regularization_losses
trainable_variables

4layers
5non_trainable_variables
	variables
6layer_regularization_losses
7layer_metrics
8metrics
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
:d2layer2/kernel
:d2layer2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
regularization_losses
trainable_variables

9layers
:non_trainable_variables
	variables
;layer_regularization_losses
<layer_metrics
=metrics
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
:d22layer3/kernel
:22layer3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
regularization_losses
trainable_variables

>layers
?non_trainable_variables
	variables
@layer_regularization_losses
Alayer_metrics
Bmetrics
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:22layer4/kernel
:2layer4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
 regularization_losses
!trainable_variables

Clayers
Dnon_trainable_variables
"	variables
Elayer_regularization_losses
Flayer_metrics
Gmetrics
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
:2layer5/kernel
:2layer5/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
н
&regularization_losses
'trainable_variables

Hlayers
Inon_trainable_variables
(	variables
Jlayer_regularization_losses
Klayer_metrics
Lmetrics
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
M0
N1
O2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
%:#	│2Adam/layer1/kernel/m
:2Adam/layer1/bias/m
$:"d2Adam/layer2/kernel/m
:d2Adam/layer2/bias/m
$:"d22Adam/layer3/kernel/m
:22Adam/layer3/bias/m
$:"22Adam/layer4/kernel/m
:2Adam/layer4/bias/m
$:"2Adam/layer5/kernel/m
:2Adam/layer5/bias/m
%:#	│2Adam/layer1/kernel/v
:2Adam/layer1/bias/v
$:"d2Adam/layer2/kernel/v
:d2Adam/layer2/bias/v
$:"d22Adam/layer3/kernel/v
:22Adam/layer3/bias/v
$:"22Adam/layer4/kernel/v
:2Adam/layer4/bias/v
$:"2Adam/layer5/kernel/v
:2Adam/layer5/bias/v
Ў2є
*__inference_sequential_layer_call_fn_89344
*__inference_sequential_layer_call_fn_89618
*__inference_sequential_layer_call_fn_89643
*__inference_sequential_layer_call_fn_89499└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨B═
 __inference__wrapped_model_89221layer1_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
т2▀
E__inference_sequential_layer_call_and_return_conditional_losses_89688
E__inference_sequential_layer_call_and_return_conditional_losses_89733
E__inference_sequential_layer_call_and_return_conditional_losses_89529
E__inference_sequential_layer_call_and_return_conditional_losses_89559└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_layer1_layer_call_fn_89743в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_layer1_layer_call_and_return_conditional_losses_89754в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_layer2_layer_call_fn_89763в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_layer2_layer_call_and_return_conditional_losses_89774в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_layer3_layer_call_fn_89783в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_layer3_layer_call_and_return_conditional_losses_89794в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_layer4_layer_call_fn_89803в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_layer4_layer_call_and_return_conditional_losses_89814в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_layer5_layer_call_fn_89823в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_layer5_layer_call_and_return_conditional_losses_89840в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▓2п
__inference_loss_fn_0_89845П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╧B╠
#__inference_signature_wrapper_89593layer1_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Щ
 __inference__wrapped_model_89221u
$%6в3
,в)
'К$
layer1_input         │
к "/к,
*
layer5 К
layer5         в
A__inference_layer1_layer_call_and_return_conditional_losses_89754]0в-
&в#
!К
inputs         │
к "%в"
К
0         
Ъ z
&__inference_layer1_layer_call_fn_89743P0в-
&в#
!К
inputs         │
к "К         б
A__inference_layer2_layer_call_and_return_conditional_losses_89774\/в,
%в"
 К
inputs         
к "%в"
К
0         d
Ъ y
&__inference_layer2_layer_call_fn_89763O/в,
%в"
 К
inputs         
к "К         dб
A__inference_layer3_layer_call_and_return_conditional_losses_89794\/в,
%в"
 К
inputs         d
к "%в"
К
0         2
Ъ y
&__inference_layer3_layer_call_fn_89783O/в,
%в"
 К
inputs         d
к "К         2б
A__inference_layer4_layer_call_and_return_conditional_losses_89814\/в,
%в"
 К
inputs         2
к "%в"
К
0         
Ъ y
&__inference_layer4_layer_call_fn_89803O/в,
%в"
 К
inputs         2
к "К         б
A__inference_layer5_layer_call_and_return_conditional_losses_89840\$%/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ y
&__inference_layer5_layer_call_fn_89823O$%/в,
%в"
 К
inputs         
к "К         7
__inference_loss_fn_0_89845в

в 
к "К ╝
E__inference_sequential_layer_call_and_return_conditional_losses_89529s
$%>в;
4в1
'К$
layer1_input         │
p 

 
к "%в"
К
0         
Ъ ╝
E__inference_sequential_layer_call_and_return_conditional_losses_89559s
$%>в;
4в1
'К$
layer1_input         │
p

 
к "%в"
К
0         
Ъ ╢
E__inference_sequential_layer_call_and_return_conditional_losses_89688m
$%8в5
.в+
!К
inputs         │
p 

 
к "%в"
К
0         
Ъ ╢
E__inference_sequential_layer_call_and_return_conditional_losses_89733m
$%8в5
.в+
!К
inputs         │
p

 
к "%в"
К
0         
Ъ Ф
*__inference_sequential_layer_call_fn_89344f
$%>в;
4в1
'К$
layer1_input         │
p 

 
к "К         Ф
*__inference_sequential_layer_call_fn_89499f
$%>в;
4в1
'К$
layer1_input         │
p

 
к "К         О
*__inference_sequential_layer_call_fn_89618`
$%8в5
.в+
!К
inputs         │
p 

 
к "К         О
*__inference_sequential_layer_call_fn_89643`
$%8в5
.в+
!К
inputs         │
p

 
к "К         н
#__inference_signature_wrapper_89593Е
$%FвC
в 
<к9
7
layer1_input'К$
layer1_input         │"/к,
*
layer5 К
layer5         