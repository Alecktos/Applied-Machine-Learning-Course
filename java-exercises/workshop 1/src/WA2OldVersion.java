import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class WA2OldVersion {

	public static void main(String[] args) {
		try {
			run();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void run() throws Exception {
		/*BufferedReader reader = new BufferedReader(
				new FileReader("../Wikipedia_70/wikipedia_70.arff"));
		Instances data = new Instances(reader);*/
		String filename = "../Wikipedia_70/wikipedia_70.arff";
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
		Instances data = arff.getData();
		System.out.println("===== Loaded dataset: " + filename + " =====");
		reader.close();

		data.setClassIndex(0);
		StringToWordVector stringToWordVector = new StringToWordVector();
		stringToWordVector.setAttributeIndices("articletype");

		FilteredClassifier classifier = new FilteredClassifier();
		classifier.setFilter(stringToWordVector);
		classifier.setClassifier(new NaiveBayesMultinomial());

		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(classifier, data, 10, new Random(1));

		System.out.println(evaluation.toSummaryString());
		System.out.println(evaluation.toClassDetailsString());
		System.out.println("===== Evaluating on filtered (training) dataset done =====");

		// setting class attribute
		/*data.setClassIndex(0);

		String[] options = weka.core.Utils.splitOptions("-R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

		StringToWordVector stringToWordVector = new StringToWordVector();                         // new instance of filter
		stringToWordVector.setOptions(options);                           // set options
		stringToWordVector.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
		stringToWordVector.setAttributeIndices("articletype");
		//Instances newData = Filter.useFilter(data, stringToWordVector);   // apply filter

		//Create classifier
		Classifier classifier = new NaiveBayesMultinomial();
		classifier.buildClassifier(newData);

		//cross validation
		Evaluation evaluation = new Evaluation(newData);
		//evaluation.crossValidateModel(naiveBayesMultinomial, newData, 10, new Random(1));
		evaluation.evaluateModel(classifier, newData);

		//System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
		// Print the algorithm summary
		System.out.println("** Naive Bayes Evaluation with Datasets **");
		System.out.println(evaluation.toSummaryString());
		System.out.print(" the expression for the input data as per alogorithm is ");
		System.out.println(classifier);
		for (int i = 0; i < newData.numInstances(); i++) {
			System.out.println(newData.instance(i));
			double index = classifier.classifyInstance(newData.instance(i));
			String className = newData.attribute(0).value((int) index);
			System.out.println(className);
		}*/


	}

}