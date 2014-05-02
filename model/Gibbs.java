package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

public class Gibbs {

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
	HashMap<String,Unit[]> tauMap;
	HashMap<String,Topic>[] docSet;
	int numOfTopics;
	int numOfDocs;
	Random rand;
	double alpha;
	
	@SuppressWarnings("unchecked")
	public Gibbs(int num,double al,int docs){
		numOfTopics=num;
		alpha=al;
		numOfDocs=docs;
		deltaMap=new Unit[docs][num];
		for(int doc=0;doc<docs;doc++){
			for(int t=0;t<num;t++){
				deltaMap[doc][t]=new Unit();
			}
		}
		tauMap=new HashMap<String,Unit[]>();
		
		docSet=(HashMap<String,Topic>[])new HashMap[numOfDocs];
		for(int t=0;t<numOfDocs;t++){
			docSet[t]=new HashMap<String,Topic>();
		}
		
		rand=new Random();
		
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
		ArrayList<String> list=new ArrayList<String>();
		for(String word:words){
			if(!word.isEmpty())
				list.add(word);
		}
		
		
		HashMap<String,Topic> docSubmap=docSet[doc];
		for(String word:list){
			docSubmap.put(word,new Topic(numOfTopics));
			if(!tauMap.containsKey(word)){
				tauMap.put(word,new Unit[numOfTopics]);
				initiateArray(tauMap.get(word));
			}
		}
		
		return list.size();
	}
	
	public void initiateParameterMaps(){
		for(int doc=0;doc<numOfDocs;doc++){
			HashMap<String,Topic> words=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(String word:words.keySet()){
				int topic=rand.nextInt(numOfTopics);
				tauMap.get(word)[topic].number++;
				deltaSubmap[topic].number++;
				words.get(word).topic=topic;
			}
		}
		updateProbs();
	}
	
	public void updateProbs(){
		int V=tauMap.size();
		for(int doc=0;doc<numOfDocs;doc++){
			Unit[] deltaSubmap=deltaMap[doc];
			int docSum=docSet[doc].size();
			for(int topic=0;topic<numOfTopics;topic++){
				Unit unit=deltaSubmap[topic];
				unit.prob=(unit.number+alpha)/(numOfTopics*alpha+docSum);
			}
		}
		
		for(int topic=0;topic<numOfTopics;topic++){
			int sum=0;
			for(String word:tauMap.keySet()){
				sum+=tauMap.get(word)[topic].number;
			}
			for(String word:tauMap.keySet()){
				Unit unit=tauMap.get(word)[topic];
				unit.prob=(unit.number+alpha)/(sum+V*alpha);
			}
		}
	}
	
	public void GibbsRecurringHelper(){
		int V=tauMap.size();
		for(int doc=0;doc<numOfDocs;doc++){
			HashMap<String,Topic> words=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(String word:words.keySet()){
				int topic=words.get(word).topic;
				tauMap.get(word)[topic].number--;
				deltaSubmap[topic].number--;
				//estimate delta probability
				double docSum=words.size();
				/*for(int t=0;t<numOfTopics;t++){
					docSum+=deltaSubmap.get(t).number;
				}*/
				
				for(int t=0;t<numOfTopics;t++){
					Unit unit=deltaSubmap[topic];
					unit.prob=(unit.number+alpha)/(docSum+alpha*numOfTopics);
				}
				//estimate tau probability
				double topicSum=0;
				for(String tWord:tauMap.keySet()){
					topicSum+=tauMap.get(tWord)[topic].number;
				}
				
				for(String tWord:tauMap.keySet()){
					Unit unit=tauMap.get(tWord)[topic];
					unit.prob=(unit.number+alpha)/(topicSum+alpha*V);
				}
				
				int newTopic=getRandomTopicForWord(doc,word);
				words.get(word).topic=newTopic;
				deltaSubmap[newTopic].number++;
				tauMap.get(word)[newTopic].number++;
			}
		}
		updateProbs();
	}
	
	public int getRandomTopicForWord(int doc,String word){
		double probSum=0;
		for(int topic=0;topic<numOfTopics;topic++){
			probSum+=tauMap.get(word)[topic].prob;
		}
		double number=rand.nextDouble()*probSum;
		int topic=0;
		for(;topic<numOfTopics;topic++){
			double prob=tauMap.get(word)[topic].prob;
			if(prob>number){
				break;
			}
			number-=prob;
		}
		return topic;
	}
	
	
	
	public double getLogLikelihood(){
		double sum=0;
		
		for(int doc=0;doc<numOfDocs;doc++){
			Unit[] deltaSubmap=deltaMap[doc];
			HashMap<String,Topic> docSubset=docSet[doc];	
			for(String word:docSubset.keySet()){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=deltaSubmap[topic].prob*tauMap.get(word)[topic].prob;
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
		do{
			prev=cur;
			GibbsRecurringHelper();
			cur=getLogLikelihood();
			System.out.println(cur);
		}while(Math.abs((cur-prev)/cur)>=precision);
		
	}
	
	public void printDeltaProb(int doc){
		Unit[] submap=deltaMap[doc];
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
		
	/*public ArrayList<ArrayList<SmoothedUnit>> getSmoothedTauProb(double alpha){
		ArrayList<ArrayList<SmoothedUnit>> smoothedMap=new ArrayList<ArrayList<SmoothedUnit>>();
		for(HashMap<String,Unit> tauSubmap:tauMap){
			double sum=0;
			ArrayList<SmoothedUnit> newList=new ArrayList<SmoothedUnit>();
			for(String word:tauSubmap.keySet()){
				sum+=tauSubmap.get(word).number;
			}
			
			for(String word:tauSubmap.keySet()){
				double prob=(tauSubmap.get(word).number+alpha)/(sum+alpha*tauSubmap.size());
				newList.add(new SmoothedUnit(word,prob));
			}
			smoothedMap.add(newList);
		}
		return smoothedMap;
	}
	
	public ArrayList<ArrayList<String>> getMostProbableWords(double alpha,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<SmoothedUnit>> smoothedMap=getSmoothedTauProb(alpha);
		for(int topic=0;topic<numOfTopics;topic++){
			PriorityQueue<SmoothedUnit> heap=new PriorityQueue<SmoothedUnit>(limit);
			ArrayList<SmoothedUnit> sublist=smoothedMap.get(topic);
			for(int i=0;i<sublist.size();i++){
				SmoothedUnit unit=sublist.get(i);
				if(heap.size()<limit){
					heap.add(unit);
					continue;
				}else if(heap.peek().prob<unit.prob){
					heap.poll();
					heap.add(unit);
				}
			}
			ArrayList<String> list=new ArrayList<String>();
			while(!heap.isEmpty()){
				list.add(0,heap.poll().word);
			}
			result.add(list);
		}
		return result;
	}
	
	public void printMostProbableWords(double alpha,int limit){
		ArrayList<ArrayList<String>> list=getMostProbableWords(alpha,limit);
		for(int topic=0;topic<numOfTopics;topic++){
			System.out.println("topic:"+topic);
			ArrayList<String> sublist=list.get(topic);
			for(String word:sublist){
				System.out.print(word+"\t");
			}
			System.out.println();
		}
	}
	*/
	public static void main(String[] args){
		Gibbs model=new Gibbs(50,0.5,1000);
		model.parseTrainingFile(args[0]);
		model.trainParameters(0.01);
		//model.printDeltaProb(16);
		//model.printMostProbableWords(1, 15);
		System.out.println("Finished");
	}
}










