/**
 * A Java class that implements a simple text learner, based on WEKA.
 * To be used with MyFilteredClassifier.java.
 * WEKA is available at: http://www.cs.waikato.ac.nz/ml/weka/
 * Copyright (C) 2013 Jose Maria Gomez Hidalgo - http://www.esp.uem.es/jmgomez
 *
 * This program is free software: you can redistribute it and/or modify
 * it for any purpose.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;

import weka.classifiers.Evaluation;
import java.util.Random;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;

/**
 * This class implements a simple text learner in Java using WEKA.
 * It loads a text dataset written in ARFF format, evaluates a classifier on it,
 * and saves the learnt model for further use.
 * @author Jose Maria Gomez Hidalgo - http://www.esp.uem.es/jmgomez
 * @see WA2TextClassification
 */
public class WA32NumericalAttributesClassification {


	/**
	 * This method loads a dataset in ARFF format. If the file does not exist, or
	 * it has a wrong format, the attribute trainData is null.
	 * @param fileName The name of the file that stores the dataset.
	 */
	public Instances loadDataset(String fileName) {
		Instances trainData = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
		return trainData;
	}

	/**
	 * This method evaluates the classifier. As recommended by WEKA documentation,
	 * the classifier is defined but not trained yet. Evaluation of previously
	 * trained classifiers can lead to unexpected results.
	 */
	public void evaluate(Instances trainData, Classifier classifier) {
		try {
			trainData.setClassIndex(1);

			Evaluation eval = new Evaluation(trainData);
			eval.crossValidateModel(classifier, trainData, 10, new Random(1));
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println("===== Evaluating on filtered (training) dataset done =====");
		}
		catch (Exception e) {
			System.out.println("Problem found when evaluating: " + e.getMessage());
		}
	}

	/**
	 * WA2OldVersion method. It is an example of the usage of this class.
	 * @param args Command-line arguments: fileData and fileModel.
	 */
	public static void main (String[] args) {
		//Classify the Iris dataset in Weka using the algorithm k-Nearest Neighbor (lazy/IBk), Decision Trees (trees/J48) and Naïve Bayes (bayes/NaiveBayes)

		WA32NumericalAttributesClassification learner = new WA32NumericalAttributesClassification();
		Instances trainData = learner.loadDataset("../Wikipedia_70/wikipedia_70.arff");

		StringToWordVector filter = new StringToWordVector();
		FilteredClassifier classifier = new FilteredClassifier();
		classifier.setFilter(filter);
		classifier.setClassifier(new NaiveBayesMultinomial());

		learner.evaluate(trainData, classifier);
	}
}