??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
{
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_36/kernel
t
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes
:	?*
dtype0
s
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:?*
dtype0
{
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_37/kernel
t
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes
:	?*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api

	0

1
2
3
 

	0

1
2
3
?
layer_metrics
non_trainable_variables
	variables

layers
metrics
regularization_losses
layer_regularization_losses
trainable_variables
 
[Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
?
layer_metrics
non_trainable_variables
	variables
 metrics

!layers
regularization_losses
"layer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
#layer_metrics
$non_trainable_variables
	variables
%metrics

&layers
regularization_losses
'layer_regularization_losses
trainable_variables
 
 
 
?
(layer_metrics
)non_trainable_variables
	variables
*metrics

+layers
regularization_losses
,layer_regularization_losses
trainable_variables
 
 

0
1
2
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
?
serving_default_dense_36_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_36_inputdense_36/kerneldense_36/biasdense_37/kerneldense_37/bias*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_86687227
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_save_86687377
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/bias*
Tin	
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__traced_restore_86687401??
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687155
dense_36_input
dense_36_86687143
dense_36_86687145
dense_37_86687148
dense_37_86687150
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_86687143dense_36_86687145*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_36_layer_call_and_return_conditional_losses_866870842"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_86687148dense_37_86687150*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_37_layer_call_and_return_conditional_losses_866871102"
 dense_37/StatefulPartitionedCall?
softmax_18/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_softmax_18_layer_call_and_return_conditional_losses_866871312
softmax_18/PartitionedCall?
IdentityIdentity#softmax_18/PartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687173

inputs
dense_36_86687161
dense_36_86687163
dense_37_86687166
dense_37_86687168
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_86687161dense_36_86687163*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_36_layer_call_and_return_conditional_losses_866870842"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_86687166dense_37_86687168*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_37_layer_call_and_return_conditional_losses_866871102"
 dense_37/StatefulPartitionedCall?
softmax_18/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_softmax_18_layer_call_and_return_conditional_losses_866871312
softmax_18/PartitionedCall?
IdentityIdentity#softmax_18/PartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687263

inputs+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd?
softmax_18/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax_18/Softmaxp
IdentityIdentitysoftmax_18/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_softmax_18_layer_call_and_return_conditional_losses_86687333

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687140
dense_36_input
dense_36_86687095
dense_36_86687097
dense_37_86687121
dense_37_86687123
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_86687095dense_36_86687097*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_36_layer_call_and_return_conditional_losses_866870842"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_86687121dense_37_86687123*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_37_layer_call_and_return_conditional_losses_866871102"
 dense_37/StatefulPartitionedCall?
softmax_18/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_softmax_18_layer_call_and_return_conditional_losses_866871312
softmax_18/PartitionedCall?
IdentityIdentity#softmax_18/PartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference__wrapped_model_86687069
dense_36_input9
5sequential_18_dense_36_matmul_readvariableop_resource:
6sequential_18_dense_36_biasadd_readvariableop_resource9
5sequential_18_dense_37_matmul_readvariableop_resource:
6sequential_18_dense_37_biasadd_readvariableop_resource
identity??
,sequential_18/dense_36/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_36_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_18/dense_36/MatMul/ReadVariableOp?
sequential_18/dense_36/MatMulMatMuldense_36_input4sequential_18/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_36/MatMul?
-sequential_18/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_18/dense_36/BiasAdd/ReadVariableOp?
sequential_18/dense_36/BiasAddBiasAdd'sequential_18/dense_36/MatMul:product:05sequential_18/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_18/dense_36/BiasAdd?
sequential_18/dense_36/ReluRelu'sequential_18/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_18/dense_36/Relu?
,sequential_18/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_37_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_18/dense_37/MatMul/ReadVariableOp?
sequential_18/dense_37/MatMulMatMul)sequential_18/dense_36/Relu:activations:04sequential_18/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_18/dense_37/MatMul?
-sequential_18/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_37/BiasAdd/ReadVariableOp?
sequential_18/dense_37/BiasAddBiasAdd'sequential_18/dense_37/MatMul:product:05sequential_18/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_18/dense_37/BiasAdd?
 sequential_18/softmax_18/SoftmaxSoftmax'sequential_18/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 sequential_18/softmax_18/Softmax~
IdentityIdentity*sequential_18/softmax_18/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference__traced_restore_86687401
file_prefix$
 assignvariableop_dense_36_kernel$
 assignvariableop_1_dense_36_bias&
"assignvariableop_2_dense_37_kernel$
 assignvariableop_3_dense_37_bias

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_36_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_36_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_37_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_37_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_signature_wrapper_86687227
dense_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_866870692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_softmax_18_layer_call_fn_86687338

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_softmax_18_layer_call_and_return_conditional_losses_866871312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_36_layer_call_fn_86687309

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_36_layer_call_and_return_conditional_losses_866870842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_sequential_18_layer_call_fn_86687289

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_18_layer_call_and_return_conditional_losses_866872012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_softmax_18_layer_call_and_return_conditional_losses_86687131

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
!__inference__traced_save_86687377
file_prefix.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_586d0bd3dc1340498698f58de67de64f/part2	
Const_1?
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*:
_input_shapes)
': :	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
?
0__inference_sequential_18_layer_call_fn_86687212
dense_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_18_layer_call_and_return_conditional_losses_866872012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_dense_36_layer_call_and_return_conditional_losses_86687084

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687245

inputs+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd?
softmax_18/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
softmax_18/Softmaxp
IdentityIdentitysoftmax_18/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_dense_36_layer_call_and_return_conditional_losses_86687300

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_sequential_18_layer_call_fn_86687184
dense_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_18_layer_call_and_return_conditional_losses_866871732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_sequential_18_layer_call_fn_86687276

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_sequential_18_layer_call_and_return_conditional_losses_866871732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_dense_37_layer_call_fn_86687328

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_37_layer_call_and_return_conditional_losses_866871102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_dense_37_layer_call_and_return_conditional_losses_86687319

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687201

inputs
dense_36_86687189
dense_36_86687191
dense_37_86687194
dense_37_86687196
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_86687189dense_36_86687191*
Tin
2*
Tout
2*(
_output_shapes
:??????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_36_layer_call_and_return_conditional_losses_866870842"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_86687194dense_37_86687196*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_37_layer_call_and_return_conditional_losses_866871102"
 dense_37/StatefulPartitionedCall?
softmax_18/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_softmax_18_layer_call_and_return_conditional_losses_866871312
softmax_18/PartitionedCall?
IdentityIdentity#softmax_18/PartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
F__inference_dense_37_layer_call_and_return_conditional_losses_86687110

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_36_input7
 serving_default_dense_36_input:0?????????>

softmax_180
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?j
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
-_default_save_signature
.__call__
*/&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_18", "layers": [{"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_18", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_18", "layers": [{"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Softmax", "config": {"name": "softmax_18", "trainable": true, "dtype": "float32", "axis": -1}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}}}}
?

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
0__call__
*1&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13]}, "stateful": false, "config": {"name": "dense_36", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.5, "maxval": 0.5, "seed": 1}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "softmax_18", "trainable": true, "dtype": "float32", "axis": -1}}
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
?
layer_metrics
non_trainable_variables
	variables

layers
metrics
regularization_losses
layer_regularization_losses
trainable_variables
.__call__
-_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
6serving_default"
signature_map
": 	?2dense_36/kernel
:?2dense_36/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?
layer_metrics
non_trainable_variables
	variables
 metrics

!layers
regularization_losses
"layer_regularization_losses
trainable_variables
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_37/kernel
:2dense_37/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
#layer_metrics
$non_trainable_variables
	variables
%metrics

&layers
regularization_losses
'layer_regularization_losses
trainable_variables
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
(layer_metrics
)non_trainable_variables
	variables
*metrics

+layers
regularization_losses
,layer_regularization_losses
trainable_variables
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
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
?2?
#__inference__wrapped_model_86687069?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
dense_36_input?????????
?2?
0__inference_sequential_18_layer_call_fn_86687212
0__inference_sequential_18_layer_call_fn_86687276
0__inference_sequential_18_layer_call_fn_86687184
0__inference_sequential_18_layer_call_fn_86687289?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687155
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687263
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687245
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687140?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dense_36_layer_call_fn_86687309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_36_layer_call_and_return_conditional_losses_86687300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_37_layer_call_fn_86687328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_37_layer_call_and_return_conditional_losses_86687319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_softmax_18_layer_call_fn_86687338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_softmax_18_layer_call_and_return_conditional_losses_86687333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
<B:
&__inference_signature_wrapper_86687227dense_36_input?
#__inference__wrapped_model_86687069x	
7?4
-?*
(?%
dense_36_input?????????
? "7?4
2

softmax_18$?!

softmax_18??????????
F__inference_dense_36_layer_call_and_return_conditional_losses_86687300]	
/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_dense_36_layer_call_fn_86687309P	
/?,
%?"
 ?
inputs?????????
? "????????????
F__inference_dense_37_layer_call_and_return_conditional_losses_86687319]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
+__inference_dense_37_layer_call_fn_86687328P0?-
&?#
!?
inputs??????????
? "???????????
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687140n	
??<
5?2
(?%
dense_36_input?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687155n	
??<
5?2
(?%
dense_36_input?????????
p 

 
? "%?"
?
0?????????
? ?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687245f	
7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
K__inference_sequential_18_layer_call_and_return_conditional_losses_86687263f	
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
0__inference_sequential_18_layer_call_fn_86687184a	
??<
5?2
(?%
dense_36_input?????????
p

 
? "???????????
0__inference_sequential_18_layer_call_fn_86687212a	
??<
5?2
(?%
dense_36_input?????????
p 

 
? "???????????
0__inference_sequential_18_layer_call_fn_86687276Y	
7?4
-?*
 ?
inputs?????????
p

 
? "???????????
0__inference_sequential_18_layer_call_fn_86687289Y	
7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
&__inference_signature_wrapper_86687227?	
I?F
? 
??<
:
dense_36_input(?%
dense_36_input?????????"7?4
2

softmax_18$?!

softmax_18??????????
H__inference_softmax_18_layer_call_and_return_conditional_losses_86687333X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_softmax_18_layer_call_fn_86687338K/?,
%?"
 ?
inputs?????????
? "??????????