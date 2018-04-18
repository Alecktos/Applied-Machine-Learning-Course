import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Main {

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
		BufferedReader reader = new BufferedReader(
				new FileReader("../Wikipedia_70/wikipedia_70.arff"));
		Instances data = new Instances(reader);


		// setting class attribute
		data.setClassIndex(0);

		String[] options = weka.core.Utils.splitOptions("-R first-last -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

		StringToWordVector stringToWordVector = new StringToWordVector();                         // new instance of filter
		stringToWordVector.setOptions(options);                           // set options
		stringToWordVector.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
		Instances newData = Filter.useFilter(data, stringToWordVector);   // apply filter


		reader.close();
	}

}