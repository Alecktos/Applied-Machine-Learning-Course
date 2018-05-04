import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import static java.lang.Math.abs;

public class WA4Regression {

	private Instances loadDataset(String fileName) {
		Instances trainData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
			trainData = arff.getData();
			//System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return trainData;
	}

	private double evaluate(Instances instances, Instance removedInstance, Classifier classifier) {
		try {


			//Evaluation eval = new Evaluation(trainData);
			//eval.crossValidateModel(classifier, trainData, 10, new Random(1));
			classifier.buildClassifier(instances);
			double expectedClass = classifier.classifyInstance(removedInstance);
			double actualClass = removedInstance.value(instances.numAttributes() - 1);

			double diff = abs(expectedClass - actualClass);
			//System.out.println("diff: " + diff);
			return diff;

			//removedInstance.setClassValue(label);
			//System.out.println(removedInstance.stringValue(6));
			//System.out.println(eval.toSummaryString());
			//System.out.println(eval.toClassDetailsString());
			//System.out.println(eval.toMatrixString());
			//System.out.println("===== Evaluating on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when evaluating: " + e.getMessage());
		}
		return 0;
	}

	private Instances getNewDataSet(Instances instances, int index) {
		/**
		 *
		 Write Java code that:
		 Iterate over all 19 training examples
		 For each iteration:
		 Remove one example from the dataset (remember to re-read the dataset each iteration otherwise all examples will be removed in iteration 19)
		 Train the k-Nearest Neighbor classifier (IBk) on the remaining 18 examples in the dataset
		 Predict the benchmark value for the training example you removed
		 Calculate the absolute difference in the predicted and actual benchmark value
		 Calculate the average absolute different over all 19 iterations
		 Experiment with different values for k. Which gives the lowest average difference?
		 Experiment with removing one attribute at a time. Is the result improved if you remove some attribute?
		 */

		Instances instances1 = new Instances(instances);
		instances1.delete(index);
		return instances1;
	}

	public void iterate() {

		for(int attributeRemoved = 0; attributeRemoved < 6; attributeRemoved++) {
			for (int k = 3; k <= 3; k++) {
				Instances trainData = loadDataset("/home/alexander/Projects/Applied-Machine-Learning-Course/GPUbenchmark/GPUbenchmark.arff");
				trainData.deleteAttributeAt(attributeRemoved);
				trainData.setClassIndex(trainData.numAttributes() - 1);
				double totalDiff = 0;
				for (int i = 0; i < 19; i++) {
					Instances instances = getNewDataSet(trainData, i);

					Instance removedInstance = trainData.get(i);

					IBk iBk = new IBk(k);

					double diff = evaluate(instances, removedInstance, iBk);
					totalDiff += diff;
				}

				System.out.println("Attribute Removed: " + attributeRemoved + " k value: " + k + " Average: " + (totalDiff / 18));
			}
		}
	}

	public static void main (String[] args) {
		//Classify the Iris dataset in Weka using the algorithm k-Nearest Neighbor (lazy/IBk), Decision Trees (trees/J48) and Naïve Bayes (bayes/NaiveBayes)

		WA4Regression learner = new WA4Regression();
		learner.iterate();


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
