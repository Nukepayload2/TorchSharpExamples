' Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
Imports System.IO
Imports System.Collections.Generic

Imports TorchSharp
Imports TorchSharp.torchvision

Imports TorchSharp.Examples
Imports TorchSharp.Examples.Utils

Imports TorchSharp.torch

Imports TorchSharp.torch.nn
Imports TorchSharp.torch.nn.functional
Imports Examples.Utils

Imports cuda = TorchSharp.torch.cuda
Imports TorchSharp.Examples.MNIST
''' <summary>
''' FGSM Attack
'''
''' Based on : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
''' </summary>
''' <remarks>
''' There are at least two interesting data sets to use with this example:
''' 
''' 1. The classic MNIST set of 60000 images of handwritten digits.
'''
'''     It is available at: http://yann.lecun.com/exdb/mnist/
'''     
''' 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
'''    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
'''    data.
'''    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
'''
''' In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
''' constant below at the folder location.
'''
''' The example is based on the PyTorch tutorial, but the results from attacking the model are very different from
''' what the tutorial article notes, at least on the machine where it was developed. There is an order-of-magnitude lower
''' drop-off in accuracy in this version. That said, when running the PyTorch tutorial on the same machine, the
''' accuracy trajectories are the same between .NET and Python. If the base convulutational model is trained
''' using Python, and then used for the FGSM attack in both .NET and Python, the drop-off trajectories are extremenly
''' close.
''' </remarks>
Public NotInheritable Class AdversarialExampleGeneration
    Private Shared ReadOnly _dataLocation As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "mnist")
    Private Shared _epochs As Integer = 4
    Private Shared _trainBatchSize As Integer = 64
    Private Shared _testBatchSize As Integer = 128
    Friend Shared Sub Run(epochs As Integer, timeout As Integer, logdir As String, dataset As String)
        _epochs = epochs

        If String.IsNullOrEmpty(dataset) Then
            dataset = "mnist"
        End If

        Dim cwd As String = Environment.CurrentDirectory

        Dim datasetPath As String = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset)

        Dim unused = torch.random.manual_seed(1)

        'var device = torch.CPU;
        Dim device = If(cuda.is_available(), VBHelper.CudaDevice, torch.CPU)
        Console.WriteLine()
        Console.WriteLine($"	Running FGSM attack with {dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.")
        Console.WriteLine()

        If device.type = DeviceType.CUDA Then
            _trainBatchSize *= 4
            _testBatchSize *= 4
            _epochs *= 4
        End If

        Console.WriteLine($"	Preparing training and test data...")

        Dim sourceDir As String = _dataLocation
        Dim targetDir As String = Path.Combine(_dataLocation, "test_data")

        Dim writer = If([String].IsNullOrEmpty(logdir), Nothing, torch.utils.tensorboard.SummaryWriter(logdir, createRunName:=True))

        If Not Directory.Exists(targetDir) Then
            Directory.CreateDirectory(targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir)
            Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir)
        End If

        Dim model1 As TorchSharp.Examples.MNIST.Model = Nothing

        Dim normImage = transforms.Normalize(New Double() {0.1307}, New Double() {0.3081}, device:=CType(device, Device))
        Using test As New MNISTReader(targetDir, "t10k", _testBatchSize, device:=device, transform:=normImage)
            Dim modelFile As String = dataset & ".model.bin"

            If Not File.Exists(modelFile) Then
                ' We need the model to be trained first, because we want to start with a trained model.
                Console.WriteLine($"
  Running MNIST on {device.type} in order to pre-train the model.")

                model1 = New TorchSharp.Examples.MNIST.Model("model", device)
                Using train As New MNISTReader(targetDir, "train", _trainBatchSize, device:=device, shuffle:=True, transform:=normImage)
                    MNIST.TrainingLoop(dataset, timeout, writer, CType(device, Device), model1, train, test)
                End Using

                Console.WriteLine("Moving on to the Adversarial model." & vbLf)
            Else
                model1 = New TorchSharp.Examples.MNIST.Model("model", torch.CPU)
                model1.load(modelFile,,)

            End If

            model1.[to](CType(device, Device))
            model1.eval()

            Dim epsilons As Double() = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5}

            For Each ε As Double In epsilons
                Dim attacked As Double = AdversarialExampleGeneration.Test(model1, NLLLoss(), ε, test, test.Size)
                Console.WriteLine($"Epsilon: {ε:F2}, accuracy: {attacked:P2}")
            Next
        End Using
    End Sub
    Private Shared Function Attack(image As Tensor, ε As Double, data_grad As Tensor) As Tensor
        Using sign = data_grad.sign()
            Dim perturbed = (image + ε * sign).clamp(0.0, 1.0)
            Return perturbed
        End Using
    End Function
    Private Shared Function Test(
        model1 As TorchSharp.Examples.MNIST.Model,
        criterion As Loss(Of Tensor, Tensor, Tensor),
        ε As Double,
        dataLoader As IEnumerable(Of (Tensor, Tensor)),
        size1 As Long) As Double
        Dim correct As Integer = 0

        For Each x In dataLoader
            Dim data = x.Item1, target = x.Item2
            Using d = torch.NewDisposeScope()
                data.requires_grad = True
                Using output = model1.forward(data)
                    Using loss = criterion.forward(output, target)
                        model1.zero_grad()
                        loss.backward()

                        Dim perturbed = Attack(data, ε, data.grad())
                        Using final = model1.forward(perturbed)
                            correct += final.argmax(1).eq(target).sum().ToInt32()
                        End Using
                    End Using
                End Using
            End Using
        Next

        Return CDbl(correct) / size1
    End Function
End Class
