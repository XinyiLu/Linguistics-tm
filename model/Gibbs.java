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
	
	ArrayList<ArrayList<Unit>> deltaMap;
	ArrayList<HashMap<String,Unit>> tauMap;
	ArrayList<HashMap<String,Topic>> docSet;
	ArrayList<Integer> tauSum;
	int numOfTopics;
	Random rand;
	
	public Gibbs(int num){
		numOfTopics=num;
		deltaMap=new ArrayList<ArrayList<Unit>>();
		tauMap=new ArrayList<HashMap<String,Unit>>();
		tauSum=new ArrayList<Integer>();
		for(int i=0;i<numOfTopics;i++){
			tauMap.add(new HashMap<String,Unit>());
			tauSum.add(0);
		}
		docSet=new ArrayList<HashMap<String,Topic>>();
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
	
	public int saveLineToSet(int doc,String line){
		String[] words=line.split(" ");
		ArrayList<String> list=new ArrayList<String>();
		for(String word:words){
			if(!word.isEmpty())
				list.add(word);
		}
		
		if(doc==docSet.size()){
			docSet.add(new HashMap<String,Topic>());
			ArrayList<Unit> deltaSubmap=new ArrayList<Unit>();
			for(int i=0;i<numOfTopics;i++){
				deltaSubmap.add(new Unit());
			}
			deltaMap.add(deltaSubmap);
			
		}
		
		HashMap<String,Topic> docSubmap=docSet.get(doc);
		for(String word:list){
			docSubmap.put(word,new Topic(numOfTopics));
			for(int i=0;i<numOfTopics;i++){
				HashMap<String,Unit> tauSubmap=tauMap.get(i);
				if(!tauSubmap.containsKey(word)){
					tauSubmap.put(word,new Unit());
				}
			}
		}
		
		return list.size();
	}
	
	public void initiateParameterMaps(){
		for(int doc=0;doc<docSet.size();doc++){
			HashMap<String,Topic> words=docSet.get(doc);
			ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
			for(String word:words.keySet()){
				int topic=rand.nextInt(numOfTopics);
				tauMap.get(topic).get(word).number++;
				deltaSubmap.get(topic).number++;
				words.get(word).topic=topic;
				tauSum.set(topic,tauSum.get(topic)+1);
			}
		}
	}
	
	public void GibbsRecurringHelper(double alpha){
		int V=tauMap.get(0).size();
		for(int doc=0;doc<docSet.size();doc++){
			HashMap<String,Topic> words=docSet.get(doc);
			ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
			for(String word:words.keySet()){
				int topic=words.get(word).topic;
				tauMap.get(topic).get(word).number--;
				deltaSubmap.get(topic).number--;
				tauSum.set(topic,tauSum.get(topic)-1);
				//estimate delta probability
				double docSum=words.size();
				/*for(int t=0;t<numOfTopics;t++){
					docSum+=deltaSubmap.get(t).number;
				}*/
				
				for(int t=0;t<numOfTopics;t++){
					Unit unit=deltaSubmap.get(topic);
					unit.prob=(unit.number+alpha)/(docSum+alpha*numOfTopics);
				}
				//estimate tau probability
				double topicSum=tauSum.get(topic);
				HashMap<String,Unit> tauSubmap=tauMap.get(topic);
				/*for(String tWord:tauSubmap.keySet()){
					topicSum+=tauSubmap.get(tWord).number;
				}*/
				
				for(String tWord:tauSubmap.keySet()){
					Unit unit=tauSubmap.get(tWord);
					unit.prob=(unit.number+alpha)/(topicSum+alpha*V);
				}
				
				int newTopic=getRandomTopicForWord(doc,word);
				words.get(word).topic=newTopic;
				deltaSubmap.get(newTopic).number++;
				tauMap.get(newTopic).get(word).number++;
				tauSum.set(newTopic,tauSum.get(newTopic)+1);
			}
		}
	}
	
	public int getRandomTopicForWord(int doc,String word){
		double probSum=0;
		for(int topic=0;topic<numOfTopics;topic++){
			probSum+=tauMap.get(topic).get(word).prob;
		}
		double number=rand.nextDouble()*probSum;
		int topic=0;
		for(;topic<numOfTopics;topic++){
			double prob=tauMap.get(topic).get(word).prob;
			if(prob>number){
				break;
			}
			number-=prob;
		}
		
		return topic;
	}
	
	
	
	public double getLogLikelihood(){
		double sum=0;
		
		for(int doc=0;doc<deltaMap.size();doc++){
			ArrayList<Unit> deltaSubmap=deltaMap.get(doc);
			HashMap<String,Topic> docSubset=docSet.get(doc);	
			for(String word:docSubset.keySet()){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=deltaSubmap.get(topic).prob*tauMap.get(topic).get(word).prob;
				}
				sum+=Math.log(wordProb);
			}
		}
		
		return sum;
	}
	
	public void trainParameters(double precision,double alpha){
		initiateParameterMaps();
		double prev=0;
		double cur=0;
		do{
			prev=cur;
			GibbsRecurringHelper(alpha);
			cur=getLogLikelihood();
			System.out.println(cur);
		}while(Math.abs((cur-prev)/cur)>=precision);
		
	}
	
	public void printDeltaProb(int doc){
		ArrayList<Unit> submap=deltaMap.get(doc);
		for(Unit unit:submap){
			System.out.println(unit.prob);
		}
	}
	
		
	public ArrayList<ArrayList<SmoothedUnit>> getSmoothedTauProb(double alpha){
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
	
	public static void main(String[] args){
		Gibbs model=new Gibbs(50);
		model.parseTrainingFile(args[0]);
		model.trainParameters(0.01,0.5);
		model.printDeltaProb(16);
		//model.printMostProbableWords(1, 15);
		System.out.println("Finished");
	}
}










