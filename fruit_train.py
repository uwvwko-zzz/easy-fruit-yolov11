from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="fruit_data.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        workers=4,        # 禁用多进程
        amp=True          # 启用自动混合精度
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()
    print("Training completed. Validation metrics:", metrics)

    # 暂时注释导出步骤，待训练完成后再执行
    # Export the model to ONNX format for deployment
    # path = model.export(format="onnx")  # Returns the path to the exported model