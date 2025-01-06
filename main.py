from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, scheduler=None, num_epochs=20):
    best_val_auc = 0.0  # 최적 validation AUC 저장
    best_epoch = 0  # 최적 validation AUC가 발생한 epoch

    for epoch in range(num_epochs):

        ### **Training Phase** ###
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # Forward Pass
            outputs = model(images)
            loss = criterion_for_train(outputs, labels)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)


        ### **Validation Phase** ###
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []
        all_val_binary_preds = []  # 이진 예측값 저장 (0 또는 1)
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.cuda(), val_labels.cuda()
                val_outputs = model(val_images)
                loss = criterion_for_test(val_outputs, val_labels)
                val_loss += loss.item()

                # 확률 값 및 이진 예측 저장
                class_1_probabilities = torch.softmax(val_outputs, dim=1)[:, 1].cpu().tolist()  # Class 1 확률
                binary_predictions = [1 if p > 0.5 else 0 for p in class_1_probabilities]
                all_val_predictions.extend(class_1_probabilities)
                all_val_binary_preds.extend(binary_predictions)
                all_val_labels.extend(val_labels.cpu().tolist())

        val_loss /= len(val_loader)

        # F1 Score, AUC, Accuracy 계산
        val_f1 = f1_score(all_val_labels, all_val_binary_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_predictions)
        val_accuracy = accuracy_score(all_val_labels, all_val_binary_preds)

        # Best Validation AUC Update
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            print(f"\nNew Best Validation AUC: {best_val_auc:.4f} at Epoch {best_epoch}")
            print(f"Validation F1 Score: {val_f1:.4f}, Validation AUC: {val_auc:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}, AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}")

        # Scheduler 업데이트
        if scheduler is not None:
            scheduler.step()
            print(f"Updated Learning Rate: {scheduler.get_last_lr()}")



    ### **Testing Phase (최종 평가)** ###
    model.eval()
    test_loss = 0.0
    all_test_labels = []
    all_test_predictions = []
    all_test_binary_preds = []
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.cuda(), test_labels.cuda()
            test_outputs = model(test_images)
            loss = criterion_for_test(test_outputs, test_labels)
            test_loss += loss.item()

            # 확률 값 및 이진 예측 저장
            class_1_probabilities = torch.softmax(test_outputs, dim=1)[:, 1].cpu().tolist()
            binary_predictions = [1 if p > 0.5 else 0 for p in class_1_probabilities]
            all_test_predictions.extend(class_1_probabilities)
            all_test_binary_preds.extend(binary_predictions)
            all_test_labels.extend(test_labels.cpu().tolist())

    test_loss /= len(test_loader)
    test_f1 = f1_score(all_test_labels, all_test_binary_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_predictions)
    test_accuracy = accuracy_score(all_test_labels, all_test_binary_preds)

    print("\nTest Results")
    print(f"Test Loss: {test_loss:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("Sample Predictions vs Actual Labels:")
    for i, (pred, actual) in enumerate(zip(all_test_binary_preds[:10], all_test_labels[:10])):
        print(f"Sample {i+1}: Predicted: {pred}, Actual: {actual}")
