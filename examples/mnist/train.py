from ml_training_template.application import TrainApplication, main
from ml_training_template.core.containers import TrainExecutor

if __name__ == "__main__":
    # app = TrainApplication()
    # app.run()

    app = TrainExecutor()
    app.core.init_resources()
    app.wire(modules=[__name__])
    # train_dataloader = app.train_dataloader().dataloader
    # valid_dataloader = app.valid_dataloader().dataloader
    # test_dataloader = app.test_dataloader().dataloader
    # model_container = app.model_container().model_container
    # trainer = app.trainer().trainer
    # print(f"train dataloader: {train_dataloader}")
    # print(f"{dir(train_dataloader)}")
    # print("====================================\n\n")
    # print(f"train dataloader: {valid_dataloader}")
    # print(f"{dir(valid_dataloader)}")
    # print("====================================\n\n")
    # print(f"train dataloader: {test_dataloader}")
    # print(f"{dir(test_dataloader)}")
    # print("====================================\n\n")
    # print(f"train dataloader: {model_container}")
    # print(f"{dir(model_container)}")
    # print("====================================\n\n")
    # print(f"train dataloader: {trainer}")
    # print(f"{dir(trainer)}")
    main()

    main()
