from keras import backend as K
K.set_learning_phase(1)

model = build_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
model.fit_generator(*)


def save_model_to_serving(model, export_version, export_path='prod_models'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'inputs': model.input}, outputs={'scores': model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))

    if tf.gfile.Exists(export_path):
        tf.gfile.DeleteRecursively(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'order_seq_predict': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()


save_model_to_serving(model, export_version='1', export_path=params.model_version_dir)
