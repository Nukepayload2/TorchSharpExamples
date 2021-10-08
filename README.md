# TorchSharp Examples

This repo holds examples and tutorials related to [TorchSharp](https://github.com/dotnet/TorchSharp), .NET-only bindings to libtorch, the engine behind PyTorch. If you are trying to familiarize yourself with TorchSharp, rather than contributing to it, this is the place to go.

Currently, the examples are the same that are also found in the TorchSharp repo. Unlike the setup in that repo, where the examples are part of the overall VS solution file and use project references to pick up the TorchSharp dependencies, in this repo, the example solution is using the publically available TorchSharp packages form NuGet. It builds faster, and is more like the 

In order to use TorchSharp, you will need both the most recent TorchSharp package, as well as one of the several libtorch-* packages that are available. The most basic one, which is used in this repository, is the libtorch-cpu package. As the name suggests, it uses a CPU backend to do training and inference.

There is also support for CUDA (10.2 and 11.1) on both Windows and Linux, and each of these combinations has its own NuGet package. If you want to train on CUDA, you need to replace references to libtorch-cpu in the solution and projects.

The examples solution should build without any modifications, either with Visual Studio, or using `dotnet build'. All of the examples build on an Nvidia GPU with 8GB of memory, while only a subset build on a GPU with 6GB.

## Structure

There are variants of all models in both C# and F#. For each of the two languages, there is a 'Models' library, and a 'XXXExamples' console app, which is what is used for batch training of the model. There is also a utility library that is written in C# only, and used from both C# and F#.

The console apps are, as mentioned, meant to be used for batch training. The command line must specify the model to be used. In the case of MNIST, there are two data sets -- the original 'MNIST' as well as the harder 'Fashion MNIST'.

The repo contains no actual data sets. You have to download them manually and, in some cases, extract the data from archives.

## Data Sets

The MNIST model uses either:

* [MNIST](http://yann.lecun.com/exdb/mnist/)
    
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

Both sets are 28x28 grayscale images, archived in .gz files.

The AlexNet, ResNet*, MobileNet, and VGG* models use the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data set. Instructions on how to download it is available in the CIFAR10 source files.

SequenceToSequence uses the [WikiText2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip) dataset. It's kept in a regular .zip file.

TextClassification uses the [AG_NEWS](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) dataset, a CSV file.

# Contributing

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

There are two main things we would like help with:

1. Adding completely new examples. File an issue and assign it to yourself, so we can track it.

2. Picking up an issue from the 'Issues' list. For example, the examples are currently set up to run on Windows, picking up data from under the 'Downloads' folder. If you have thoughts on the best way to do this on MacOS or Linux, please help with that.

If you add a new example, please adjust it to work on a mainstream CUDA processor. This means making sure that it builds on an 8GB processor, with sufficient invocations of the garbage collector.