import weka.core.Instances;

public class WA4Regression {

	public static void main (String[] args) {
		//Classify the Iris dataset in Weka using the algorithm k-Nearest Neighbor (lazy/IBk), Decision Trees (trees/J48) and Naïve Bayes (bayes/NaiveBayes)

		WA32NumericalAttributesClassification learner = new WA32NumericalAttributesClassification();
		Instances trainData = learner.loadDataset("/home/alexander/Projects/Applied-Machine-Learning-Course/Iris/iris.arff");

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
