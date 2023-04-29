' Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
Imports System.IO
Imports System.Linq
Imports System.Collections.Generic
Imports System.Diagnostics

Imports TorchSharp
Imports TorchSharp.torchvision

Imports TorchSharp.Examples
Imports TorchSharp.Examples.Utils

Imports TorchSharp.torch

Imports TorchSharp.torch.nn
Imports TorchSharp.torch.nn.functional
Imports cuda = TorchSharp.torch.cuda
Imports Examples.Utils

''' <summary>
''' Driver for various models trained and evaluated on the CIFAR10 small (32x32) color image data set.
''' </summary>
''' <remarks>
''' The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
''' Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
''' the '_dataLocation' definition below to point at the right folder.
'''
''' Note: so far, CIFAR10 is supported, but not CIFAR100.
''' </remarks>
NotInheritable Class CIFAR10
    Private Shared ReadOnly _dataset As String = "CIFAR10"
    Private Shared ReadOnly _dataLocation As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", _dataset)
    Private Shared _trainBatchSize As Integer = 64
    Private Shared _testBatchSize As Integer = 128
    Private Shared ReadOnly _logInterval As Integer = 25
    Private Shared ReadOnly _numClasses As Integer = 10
    Friend Shared Sub Run(epochs As Integer, timeout As Integer, logdir As String, modelName As String)
        torch.random.manual_seed(1)
        ' This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
        ' It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
        Dim device = If(cuda.is_available(), VBHelper.CudaDevice, torch.CPU)

        If device.type = DeviceType.CUDA Then
            _trainBatchSize *= 8
            _testBatchSize *= 8
        End If

        Console.WriteLine()
        Console.WriteLine($"	Running {modelName} with {_dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.")
        Console.WriteLine()

        Dim writer = If([String].IsNullOrEmpty(logdir), Nothing, torch.utils.tensorboard.SummaryWriter(logdir, createRunName:=True))

        Dim sourceDir As String = _dataLocation
        Dim targetDir As String = Path.Combine(_dataLocation, "test_data")

        If Not Directory.Exists(targetDir) Then
            Directory.CreateDirectory(targetDir)
            Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir)
        End If

        Console.WriteLine($"	Creating the model...")

        Dim model As [Module](Of Tensor, Tensor) = Nothing

        Select Case modelName.ToLower()
            Case "alexnet"
                model = New AlexNet(modelName, _numClasses, device)
            Case "mobilenet"
                model = New MobileNet(modelName, _numClasses, device)
            Case "vgg11", "vgg13", "vgg16", "vgg19"
                model = New VGG(modelName, _numClasses, device)
            Case "resnet18"
                model = ResNet.ResNet18(_numClasses, device)
            Case "resnet34"
                _testBatchSize \= 4
                model = ResNet.ResNet34(_numClasses, device)
            Case "resnet50"
                _trainBatchSize \= 6
                _testBatchSize \= 8
                model = ResNet.ResNet50(_numClasses, device)
            Case "resnet101"
                _trainBatchSize \= 6
                _testBatchSize \= 8
                model = ResNet.ResNet101(_numClasses, device)
            Case "resnet152"
                _testBatchSize \= 4
                model = ResNet.ResNet152(_numClasses, device)
        End Select

        Dim hflip = transforms.HorizontalFlip()
        Dim gray = transforms.Grayscale(3)
        Dim rotate1 = transforms.Rotate(90)
        Dim contrast = transforms.AdjustContrast(1.25)

        Console.WriteLine($"	Preparing training and test data...")
        Console.WriteLine()
        Using train As New CIFARReader(targetDir, False, _trainBatchSize, shuffle:=True, device:=device, transforms:=New ITransform() {})
            Using test As New CIFARReader(targetDir, True, _testBatchSize, device:=device)
                Using optimizer = torch.optim.Adam(model.parameters(), 0.001)
                    Dim totalSW As New Stopwatch
                    totalSW.Start()

                    For epoch = 1 To epochs
                        Dim epchSW As New Stopwatch
                        epchSW.Start()

                        Dim loss = NLLLoss()

                        CIFAR10.Train(model, optimizer, loss, train.Data(), epoch, _trainBatchSize, train.Size)
                        CIFAR10.Test(model, loss, writer, modelName.ToLower(), test.Data(), epoch, test.Size)

                        epchSW.[Stop]()
                        Console.WriteLine($"Elapsed time for this epoch: {epchSW.Elapsed.TotalSeconds} s.")

                        If totalSW.Elapsed.TotalSeconds > timeout Then
                            Exit For
                        End If
                    Next

                    totalSW.[Stop]()
                    Console.WriteLine($"Elapsed training time: {totalSW.Elapsed} s.")
                End Using
            End Using
        End Using

        model.Dispose()
    End Sub
    Private Shared Sub Train(
        model As [Module](Of Tensor, Tensor),
        optimizer As torch.optim.Optimizer,
        loss As Loss(Of Tensor, Tensor, Tensor),
        dataLoader As IEnumerable(Of (Tensor, Tensor)),
        epoch As Integer,
        batchSize As Long,
        size1 As Long)
        model.train()

        Dim batchId As Integer = 1
        Dim total As Long = 0
        Dim correct As Long = 0

        Console.WriteLine($"Epoch: {epoch}...")

        For Each x In dataLoader
            Dim data = x.Item1, target = x.Item2
            Using d = torch.NewDisposeScope()
                optimizer.zero_grad()

                Dim prediction = model.forward(data)
                Dim lsm = log_softmax(prediction, 1)
                Dim output = loss.forward(lsm, target)

                output.backward()

                optimizer.[step]()

                total += target.shape(0)

                correct += prediction.argmax(1).eq(target).sum().ToInt64()

                If batchId Mod _logInterval = 0 Then
                    Dim count = Math.Min(batchId * batchSize, size1)
                    Console.WriteLine($"
Train: epoch {epoch} [{count} / {size1}] Loss: {output.ToSingle().ToString("0.000000")} | Accuracy: {(CSng(correct) / total).ToString("0.000000") }")
                End If

                batchId += 1
            End Using
        Next

    End Sub
    Private Shared Sub Test(
        model As [Module](Of Tensor, Tensor),
        loss As Loss(Of Tensor, Tensor, Tensor),
        writer As TorchSharp.Modules.SummaryWriter,
        modelName As String,
        dataLoader As IEnumerable(Of (Tensor, Tensor)),
        epoch As Integer,
        size1 As Long)
        model.eval()

        Dim testLoss As Double = 0
        Dim correct As Long = 0
        Dim batchCount As Integer = 0

        For Each x In dataLoader
            Dim data = x.Item1, target = x.Item2
            Using d = torch.NewDisposeScope()
                Dim prediction = model.forward(data)
                Dim lsm = log_softmax(prediction, 1)
                Dim output = loss.forward(lsm, target)

                testLoss += output.ToSingle()
                batchCount += 1

                correct += prediction.argmax(1).eq(target).sum().ToInt64()
            End Using
        Next

        Console.WriteLine($"
Test set: Average loss {(testLoss / batchCount).ToString("0.0000")} | Accuracy {(CSng(correct) / size1).ToString("0.0000")}")

        If writer IsNot Nothing Then
            writer.add_scalar($"{modelName}/loss", CSng((testLoss / batchCount)), epoch)
            writer.add_scalar($"{modelName}/accuracy", CSng(correct) / size1, epoch)
        End If
    End Sub
End Class
