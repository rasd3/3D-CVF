# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/optimizer.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/optimizer.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1dsecond/protos/optimizer.proto\x12\rsecond.protos\"\xa5\x02\n\tOptimizer\x12=\n\x12rms_prop_optimizer\x18\x01 \x01(\x0b\x32\x1f.second.protos.RMSPropOptimizerH\x00\x12>\n\x12momentum_optimizer\x18\x02 \x01(\x0b\x32 .second.protos.MomentumOptimizerH\x00\x12\x36\n\x0e\x61\x64\x61m_optimizer\x18\x03 \x01(\x0b\x32\x1c.second.protos.AdamOptimizerH\x00\x12\x1a\n\x12use_moving_average\x18\x04 \x01(\x08\x12\x1c\n\x14moving_average_decay\x18\x05 \x01(\x02\x12\x1a\n\x12\x66ixed_weight_decay\x18\x06 \x01(\x08\x42\x0b\n\toptimizer\"\x9e\x01\n\x10RMSPropOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.second.protos.LearningRate\x12 \n\x18momentum_optimizer_value\x18\x02 \x01(\x02\x12\r\n\x05\x64\x65\x63\x61y\x18\x03 \x01(\x02\x12\x0f\n\x07\x65psilon\x18\x04 \x01(\x02\x12\x14\n\x0cweight_decay\x18\x05 \x01(\x02\"\x7f\n\x11MomentumOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.second.protos.LearningRate\x12 \n\x18momentum_optimizer_value\x18\x02 \x01(\x02\x12\x14\n\x0cweight_decay\x18\x03 \x01(\x02\"j\n\rAdamOptimizer\x12\x32\n\rlearning_rate\x18\x01 \x01(\x0b\x32\x1b.second.protos.LearningRate\x12\x14\n\x0cweight_decay\x18\x02 \x01(\x02\x12\x0f\n\x07\x61msgrad\x18\x03 \x01(\x08\"\xb9\x01\n\x0cLearningRate\x12\x30\n\x0bmulti_phase\x18\x01 \x01(\x0b\x32\x19.second.protos.MultiPhaseH\x00\x12,\n\tone_cycle\x18\x02 \x01(\x0b\x32\x17.second.protos.OneCycleH\x00\x12\x38\n\x0fmanual_stepping\x18\x03 \x01(\x0b\x32\x1d.second.protos.ManualSteppingH\x00\x42\x0f\n\rlearning_rate\"U\n\x11LearningRatePhase\x12\r\n\x05start\x18\x01 \x01(\x02\x12\x13\n\x0blambda_func\x18\x02 \x01(\t\x12\x1c\n\x14momentum_lambda_func\x18\x03 \x01(\t\">\n\nMultiPhase\x12\x30\n\x06phases\x18\x01 \x03(\x0b\x32 .second.protos.LearningRatePhase\"O\n\x08OneCycle\x12\x0e\n\x06lr_max\x18\x01 \x01(\x02\x12\x0c\n\x04moms\x18\x02 \x03(\x02\x12\x12\n\ndiv_factor\x18\x03 \x01(\x02\x12\x11\n\tpct_start\x18\x04 \x01(\x02\"3\n\x0eManualStepping\x12\x12\n\nboundaries\x18\x01 \x03(\x02\x12\r\n\x05rates\x18\x02 \x03(\x02\x62\x06proto3')
)




_OPTIMIZER = _descriptor.Descriptor(
  name='Optimizer',
  full_name='second.protos.Optimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rms_prop_optimizer', full_name='second.protos.Optimizer.rms_prop_optimizer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer', full_name='second.protos.Optimizer.momentum_optimizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='adam_optimizer', full_name='second.protos.Optimizer.adam_optimizer', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_moving_average', full_name='second.protos.Optimizer.use_moving_average', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='moving_average_decay', full_name='second.protos.Optimizer.moving_average_decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fixed_weight_decay', full_name='second.protos.Optimizer.fixed_weight_decay', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='optimizer', full_name='second.protos.Optimizer.optimizer',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=49,
  serialized_end=342,
)


_RMSPROPOPTIMIZER = _descriptor.Descriptor(
  name='RMSPropOptimizer',
  full_name='second.protos.RMSPropOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='second.protos.RMSPropOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='second.protos.RMSPropOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decay', full_name='second.protos.RMSPropOptimizer.decay', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='second.protos.RMSPropOptimizer.epsilon', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='second.protos.RMSPropOptimizer.weight_decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=345,
  serialized_end=503,
)


_MOMENTUMOPTIMIZER = _descriptor.Descriptor(
  name='MomentumOptimizer',
  full_name='second.protos.MomentumOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='second.protos.MomentumOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum_optimizer_value', full_name='second.protos.MomentumOptimizer.momentum_optimizer_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='second.protos.MomentumOptimizer.weight_decay', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=505,
  serialized_end=632,
)


_ADAMOPTIMIZER = _descriptor.Descriptor(
  name='AdamOptimizer',
  full_name='second.protos.AdamOptimizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='learning_rate', full_name='second.protos.AdamOptimizer.learning_rate', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='weight_decay', full_name='second.protos.AdamOptimizer.weight_decay', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='amsgrad', full_name='second.protos.AdamOptimizer.amsgrad', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=634,
  serialized_end=740,
)


_LEARNINGRATE = _descriptor.Descriptor(
  name='LearningRate',
  full_name='second.protos.LearningRate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='multi_phase', full_name='second.protos.LearningRate.multi_phase', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='one_cycle', full_name='second.protos.LearningRate.one_cycle', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='manual_stepping', full_name='second.protos.LearningRate.manual_stepping', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='learning_rate', full_name='second.protos.LearningRate.learning_rate',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=743,
  serialized_end=928,
)


_LEARNINGRATEPHASE = _descriptor.Descriptor(
  name='LearningRatePhase',
  full_name='second.protos.LearningRatePhase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start', full_name='second.protos.LearningRatePhase.start', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lambda_func', full_name='second.protos.LearningRatePhase.lambda_func', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum_lambda_func', full_name='second.protos.LearningRatePhase.momentum_lambda_func', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=930,
  serialized_end=1015,
)


_MULTIPHASE = _descriptor.Descriptor(
  name='MultiPhase',
  full_name='second.protos.MultiPhase',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='phases', full_name='second.protos.MultiPhase.phases', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1017,
  serialized_end=1079,
)


_ONECYCLE = _descriptor.Descriptor(
  name='OneCycle',
  full_name='second.protos.OneCycle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lr_max', full_name='second.protos.OneCycle.lr_max', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='moms', full_name='second.protos.OneCycle.moms', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='div_factor', full_name='second.protos.OneCycle.div_factor', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pct_start', full_name='second.protos.OneCycle.pct_start', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1081,
  serialized_end=1160,
)


_MANUALSTEPPING = _descriptor.Descriptor(
  name='ManualStepping',
  full_name='second.protos.ManualStepping',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='boundaries', full_name='second.protos.ManualStepping.boundaries', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rates', full_name='second.protos.ManualStepping.rates', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1162,
  serialized_end=1213,
)

_OPTIMIZER.fields_by_name['rms_prop_optimizer'].message_type = _RMSPROPOPTIMIZER
_OPTIMIZER.fields_by_name['momentum_optimizer'].message_type = _MOMENTUMOPTIMIZER
_OPTIMIZER.fields_by_name['adam_optimizer'].message_type = _ADAMOPTIMIZER
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['rms_prop_optimizer'])
_OPTIMIZER.fields_by_name['rms_prop_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['momentum_optimizer'])
_OPTIMIZER.fields_by_name['momentum_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_OPTIMIZER.oneofs_by_name['optimizer'].fields.append(
  _OPTIMIZER.fields_by_name['adam_optimizer'])
_OPTIMIZER.fields_by_name['adam_optimizer'].containing_oneof = _OPTIMIZER.oneofs_by_name['optimizer']
_RMSPROPOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_MOMENTUMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_ADAMOPTIMIZER.fields_by_name['learning_rate'].message_type = _LEARNINGRATE
_LEARNINGRATE.fields_by_name['multi_phase'].message_type = _MULTIPHASE
_LEARNINGRATE.fields_by_name['one_cycle'].message_type = _ONECYCLE
_LEARNINGRATE.fields_by_name['manual_stepping'].message_type = _MANUALSTEPPING
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['multi_phase'])
_LEARNINGRATE.fields_by_name['multi_phase'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['one_cycle'])
_LEARNINGRATE.fields_by_name['one_cycle'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_LEARNINGRATE.oneofs_by_name['learning_rate'].fields.append(
  _LEARNINGRATE.fields_by_name['manual_stepping'])
_LEARNINGRATE.fields_by_name['manual_stepping'].containing_oneof = _LEARNINGRATE.oneofs_by_name['learning_rate']
_MULTIPHASE.fields_by_name['phases'].message_type = _LEARNINGRATEPHASE
DESCRIPTOR.message_types_by_name['Optimizer'] = _OPTIMIZER
DESCRIPTOR.message_types_by_name['RMSPropOptimizer'] = _RMSPROPOPTIMIZER
DESCRIPTOR.message_types_by_name['MomentumOptimizer'] = _MOMENTUMOPTIMIZER
DESCRIPTOR.message_types_by_name['AdamOptimizer'] = _ADAMOPTIMIZER
DESCRIPTOR.message_types_by_name['LearningRate'] = _LEARNINGRATE
DESCRIPTOR.message_types_by_name['LearningRatePhase'] = _LEARNINGRATEPHASE
DESCRIPTOR.message_types_by_name['MultiPhase'] = _MULTIPHASE
DESCRIPTOR.message_types_by_name['OneCycle'] = _ONECYCLE
DESCRIPTOR.message_types_by_name['ManualStepping'] = _MANUALSTEPPING
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Optimizer = _reflection.GeneratedProtocolMessageType('Optimizer', (_message.Message,), dict(
  DESCRIPTOR = _OPTIMIZER,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.Optimizer)
  ))
_sym_db.RegisterMessage(Optimizer)

RMSPropOptimizer = _reflection.GeneratedProtocolMessageType('RMSPropOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _RMSPROPOPTIMIZER,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.RMSPropOptimizer)
  ))
_sym_db.RegisterMessage(RMSPropOptimizer)

MomentumOptimizer = _reflection.GeneratedProtocolMessageType('MomentumOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _MOMENTUMOPTIMIZER,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.MomentumOptimizer)
  ))
_sym_db.RegisterMessage(MomentumOptimizer)

AdamOptimizer = _reflection.GeneratedProtocolMessageType('AdamOptimizer', (_message.Message,), dict(
  DESCRIPTOR = _ADAMOPTIMIZER,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.AdamOptimizer)
  ))
_sym_db.RegisterMessage(AdamOptimizer)

LearningRate = _reflection.GeneratedProtocolMessageType('LearningRate', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATE,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.LearningRate)
  ))
_sym_db.RegisterMessage(LearningRate)

LearningRatePhase = _reflection.GeneratedProtocolMessageType('LearningRatePhase', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATEPHASE,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.LearningRatePhase)
  ))
_sym_db.RegisterMessage(LearningRatePhase)

MultiPhase = _reflection.GeneratedProtocolMessageType('MultiPhase', (_message.Message,), dict(
  DESCRIPTOR = _MULTIPHASE,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.MultiPhase)
  ))
_sym_db.RegisterMessage(MultiPhase)

OneCycle = _reflection.GeneratedProtocolMessageType('OneCycle', (_message.Message,), dict(
  DESCRIPTOR = _ONECYCLE,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.OneCycle)
  ))
_sym_db.RegisterMessage(OneCycle)

ManualStepping = _reflection.GeneratedProtocolMessageType('ManualStepping', (_message.Message,), dict(
  DESCRIPTOR = _MANUALSTEPPING,
  __module__ = 'second.protos.optimizer_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.ManualStepping)
  ))
_sym_db.RegisterMessage(ManualStepping)


# @@protoc_insertion_point(module_scope)
