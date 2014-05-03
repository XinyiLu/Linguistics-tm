package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Random;

public class Gibbs2 {

	class Unit{
		double prob;
		double number;
		
		public Unit(){
			prob=0;
			number=0;
		}
		
		public Unit(double p,double n){
			prob=p;
			number=n;
		}
	}
	
	class Topic{
		int topic;
		public Topic(int t){
			topic=t;
		}
	}
	
	@SuppressWarnings("rawtypes")
	class SmoothedUnit implements Comparable{
		String word;
		double prob;

		public SmoothedUnit(String w,double p){
			word=w;
			prob=p;
		}
		@Override
		public int compareTo(Object arg0) {
			SmoothedUnit unit=(SmoothedUnit)arg0;
			if(unit.prob<prob){
				return 1;
			}else if(unit.prob==prob){
				return 0;
			}else{
				return -1;
			}
		}
	}
	
	Unit[][] deltaMap;
	HashMap<String,Unit>[] tauMap;
	HashMap<String,Topic>[] docSet;
	int numOfTopics;
	int numOfDocs;
	Random rand;
	double alpha;
	int[] tauSum;
	
	@SuppressWarnings("unchecked")
	public Gibbs2(int num,double al,int docs){
		numOfTopics=num;
		alpha=al;
		numOfDocs=docs;
		deltaMap=new Unit[docs][num];
		tauMap=(HashMap<String,Unit>[])new HashMap[numOfTopics];
		docSet=(HashMap<String,Topic>[])new HashMap[numOfDocs];
		
		for(int doc=0;doc<docs;doc++){
			docSet[doc]=new HashMap<String,Topic>();
			for(int t=0;t<num;t++){
				deltaMap[doc][t]=new Unit();
			}
		}
		
		for(int t=0;t<numOfTopics;t++){
			tauMap[t]=new HashMap<String,Unit>();
		}
		
		rand=new Random();
		tauSum=new int[numOfTopics];
	}
	
	public void parseTrainingFile(String fileName){
		
		try {
			BufferedReader reader=new BufferedReader(new InputStreamReader(new FileInputStream(fileName),"ISO-8859-1"));
			String line=null;
			//each time we read a line, count its words
			int doc=0;
			while((line=reader.readLine())!=null){
				while(line.isEmpty()){
					line=reader.readLine();
				}
				int count=Integer.parseInt(line);
				while(count>0){
					line=reader.readLine();
					count-=saveLineToSet(doc,line);
				}
				doc++;
			}
			//close the buffered reader
			reader.close();

		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void initiateArray(Unit[] list){
		for(int i=0;i<list.length;i++){
			list[i]=new Unit();
		}
	}
	
	
	public int saveLineToSet(int doc,String line){
		String[] words=line.split(" ");
		/*ArrayList<String> list=new ArrayList<String>();
		for(String word:words){
			if(!word.isEmpty())
				list.add(word);
		}*/
		
		
		HashMap<String,Topic> docSubmap=docSet[doc];
		for(String word:words){
			docSubmap.put(word,new Topic(numOfTopics));
			for(int t=0;t<numOfTopics;t++){
				HashMap<String,Unit> tempMap=tauMap[t];
				if(!tempMap.containsKey(word))
					tempMap.put(word,new Unit());
			}
		}
		
		return words.length-1;
	}
	
	public void initiateParameterMaps(){
		for(int doc=0;doc<numOfDocs;doc++){
			HashMap<String,Topic> words=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(String word:words.keySet()){
				int topic=rand.nextInt(numOfTopics);
				tauMap[topic].get(word).number++;
				deltaSubmap[topic].number++;
				words.get(word).topic=topic;
				tauSum[topic]++;
			}
		}
		
		updateProbs();
	}
	
	public void updateProbs(){
		int V=tauMap[0].size();
		for(int doc=0;doc<numOfDocs;doc++){
			Unit[] deltaSubmap=deltaMap[doc];
			int docSum=docSet[doc].size();
			for(int topic=0;topic<numOfTopics;topic++){
				Unit unit=deltaSubmap[topic];
				unit.prob=(unit.number+alpha)/(numOfTopics*alpha+docSum);
			}
		}
		
		for(int topic=0;topic<numOfTopics;topic++){
			int sum=tauSum[topic];
			HashMap<String,Unit> tauSubmap=tauMap[topic];
			for(String word:tauSubmap.keySet()){
				Unit unit=tauSubmap.get(word);
				unit.prob=(unit.number+alpha)/(sum+V*alpha);
			}
		}
	}
	
	public void updateTopicProbs(int doc,int topic){
		int V=tauMap[0].size();
		double docSum=docSet[doc].size();
		Unit[] deltaSubmap=deltaMap[doc];
		for(int t=0;t<numOfTopics;t++){
			Unit unit=deltaSubmap[topic];
			unit.prob=(unit.number+alpha)/(docSum+alpha*numOfTopics);
		}
		//estimate tau probability
		double topicSum=tauSum[topic];
		HashMap<String,Unit> tauSubmap=tauMap[topic];
		for(String tWord:tauSubmap.keySet()){
			Unit unit=tauSubmap.get(tWord);
			unit.prob=(unit.number+alpha)/(topicSum+alpha*V);
		}
	}
	
	public void GibbsRecurringHelper(){
		for(int doc=0;doc<numOfDocs;doc++){
			HashMap<String,Topic> words=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(String word:words.keySet()){
				int topic=words.get(word).topic;
				tauMap[topic].get(word).number--;
				deltaSubmap[topic].number--;
				tauSum[topic]--;
				updateTopicProbs(doc,topic);
				int newTopic=getRandomTopicForWord(doc,word);
				words.get(word).topic=newTopic;
				deltaSubmap[newTopic].number++;
				tauMap[newTopic].get(word).number++;
				tauSum[newTopic]++;
				updateTopicProbs(doc,newTopic);
			}
		}
	}
	
	public int getRandomTopicForWord(int doc,String word){
		double probSum=0;
		
		Unit[] deltalist=deltaMap[doc];
		for(int topic=0;topic<numOfTopics;topic++){
			probSum+=tauMap[topic].get(word).prob*deltalist[topic].prob;
		}
		double number=rand.nextDouble()*probSum;
		int topic=0;
		for(;topic<numOfTopics;topic++){
			double prob=tauMap[topic].get(word).prob*deltalist[topic].prob;
			if(prob>number){
				break;
			}
			number-=prob;
		}
		return topic;
	}
	
	
	
	public double getLogLikelihood(){
		updateProbs();
		double sum=0;
		
		for(int doc=0;doc<numOfDocs;doc++){
			Unit[] deltaSubmap=deltaMap[doc];
			HashMap<String,Topic> docSubset=docSet[doc];	
			for(String word:docSubset.keySet()){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=deltaSubmap[topic].prob*tauMap[topic].get(word).prob;
				}
				sum+=Math.log(wordProb);
			}
		}
		
		return sum;
	}
	
	public void trainParameters(double precision){
		initiateParameterMaps();
		double prev=0;
		double cur=0;
		int count=0;
		count++;
		do{
			prev=cur;
			GibbsRecurringHelper();
			cur=getLogLikelihood();
			System.out.println(cur);
		}while(count<50||Math.abs((cur-prev)/cur)>=precision);
		
	}
	
	public void printDeltaProb(int doc){
		Unit[] submap=deltaMap[doc];
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
	
	public static void main(String[] args){
		Gibbs2 model=new Gibbs2(50,0.5,1000);
		model.parseTrainingFile(args[0]);
		System.out.println("File Parsed");
		model.trainParameters(0.01);
		model.printDeltaProb(16);
		//model.printMostProbableWords(1, 15);
		System.out.println("Finished");
	}
}





