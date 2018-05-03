import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class WA4Regression {

	public Instances loadDataset(String fileName) {
		Instances trainData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return trainData;
	}

	private void iterate() {

	}

	public static void main (String[] args) {
		//Classify the Iris dataset in Weka using the algorithm k-Nearest Neighbor (lazy/IBk), Decision Trees (trees/J48) and Naïve Bayes (bayes/NaiveBayes)

		WA4Regression learner = new WA4Regression();
		Instances trainData = learner.loadDataset("/home/alexander/Projects/Applied-Machine-Learning-Course/GPUbenchmark/GPUbenchmark.arff");

		//Naive Bayes
		//NaiveBayes naiveBayes = new NaiveBayes();
		//learner.evaluate(trainData, naiveBayes);

		//Decision tree
		//J48 j48 = new J48();
		//learner.evaluate(trainData, j48);

		//IBk iBk = new IBk();
		///learner.evaluate(trainData, iBk);
	}
}
