import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class WA2 {


	public void run() {
		Instances instances = loadDataset("/home/alexander/Projects/Applied-Machine-Learning-Course/diabetes/diabetes.arff");

		RandomForest randomForest = new RandomForest();
		//MultilayerPerceptron multilayerPerceptron = new MultilayerPerceptron();
		evaluate(randomForest, instances);
	}

	private Instances loadDataset(String fileName) {
		Instances trainData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		}
		catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return trainData;
	}

	private void evaluate(Classifier classifier, Instances trainData) {
		try {
			trainData.setClassIndex(8);

			StringBuffer predsBuffer = new StringBuffer();
			PlainText plainText = new PlainText();
			plainText.setHeader(trainData);
			plainText.setBuffer(predsBuffer);

			Evaluation eval = new Evaluation(trainData);
			eval.crossValidateModel(classifier, trainData, 10, new Random(1), plainText);
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());
			System.out.println("===== Evaluating on filtered (training) dataset done =====");
			System.out.println(predsBuffer.toString());
		} catch (Exception e) {
			System.out.println("=====  Wrong result =====");
		}
	}



}
