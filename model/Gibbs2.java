package model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

import model.PLSA.SmoothedUnit;
import model.PLSA.Unit;

public class Gibbs2 {

	class Unit{
		double prob;
		int number;
		
		public Unit(){
			prob=0;
			number=0;
		}
		
		public Unit(double p,int n){
			prob=p;
			number=n;
		}
	}
	
	class TopicUnit{
		String word;
		int topic;
		public TopicUnit(String w,int t){
			word=w;
			topic=t;
		}
		
		public TopicUnit(String w){
			word=w;
			topic=numOfTopics;
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
	ArrayList<TopicUnit>[] docSet;
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
		docSet=(ArrayList<TopicUnit>[])new ArrayList[numOfDocs];
		
		for(int doc=0;doc<docs;doc++){
			docSet[doc]=new ArrayList<TopicUnit>();
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
		ArrayList<String> list=new ArrayList<String>();
		for(String word:words){
			if(!word.isEmpty())
				list.add(word);
		}

		ArrayList<TopicUnit> docSubmap=docSet[doc];
		for(String word:list){
			docSubmap.add(new TopicUnit(word));
			for(int t=0;t<numOfTopics;t++){
				HashMap<String,Unit> tempMap=tauMap[t];
				if(!tempMap.containsKey(word))
					tempMap.put(word,new Unit());
			}
		}
		
		return list.size();
	}
	
	public void initiateParameterMaps(){
		for(int doc=0;doc<numOfDocs;doc++){
			ArrayList<TopicUnit> wordsMap=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(TopicUnit unit:wordsMap){				
				int topic=rand.nextInt(numOfTopics);
				tauMap[topic].get(unit.word).number++;
				deltaSubmap[topic].number++;
				unit.topic=topic;
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
		int docSize=docSet[doc].size();
		Unit[] deltaSubmap=deltaMap[doc];
		for(int t=0;t<numOfTopics;t++){
			Unit unit=deltaSubmap[topic];
			unit.prob=(unit.number+alpha)/(docSize+alpha*numOfTopics);
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
			ArrayList<TopicUnit> words=docSet[doc];
			Unit[] deltaSubmap=deltaMap[doc];
			for(TopicUnit unit:words){
				int topic=unit.topic;
				tauMap[topic].get(unit.word).number--;
				deltaSubmap[topic].number--;
				tauSum[topic]--;
				updateTopicProbs(doc,topic);
				int newTopic=getRandomTopicForWord(doc,unit.word);
				unit.topic=newTopic;
				deltaSubmap[newTopic].number++;
				tauMap[newTopic].get(unit.word).number++;
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
		double sum=0;
		
		for(int doc=0;doc<numOfDocs;doc++){
			Unit[] deltaSubmap=deltaMap[doc];
			ArrayList<TopicUnit> docSubset=docSet[doc];	
			for(TopicUnit unit:docSubset){
				double wordProb=0;
				for(int topic=0;topic<numOfTopics;topic++){
					wordProb+=deltaSubmap[topic].prob*tauMap[topic].get(unit.word).prob;
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
		do{
			count++;
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
	
	@SuppressWarnings("unchecked")
	public ArrayList<SmoothedUnit>[] getSmoothedTauProb(double theta){
		ArrayList<SmoothedUnit>[] smoothedMap=new ArrayList[numOfTopics];
		for(int t=0;t<numOfTopics;t++){
			smoothedMap[t]=new ArrayList<SmoothedUnit>();
		}
		
		
		HashMap<String,Unit> submap=tauMap[0];
		for(String word:submap.keySet()){
			double sum=0;
			for(int topic=0;topic<numOfTopics;topic++){
				sum+=tauMap[topic].get(word).number;
			}
			
			for(int topic=0;topic<numOfTopics;topic++){
				double tempProb=(tauMap[topic].get(word).number+theta)/(sum+theta*numOfTopics);
				smoothedMap[topic].add(new SmoothedUnit(word,tempProb));
			}
		}
		
		return smoothedMap;
	}
	
	public ArrayList<ArrayList<String>> getMostProbableWords(double theta,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<SmoothedUnit>[] smoothedMap=getSmoothedTauProb(theta);
		for(int topic=0;topic<numOfTopics;topic++){
			PriorityQueue<SmoothedUnit> heap=new PriorityQueue<SmoothedUnit>(limit);
			ArrayList<SmoothedUnit> sublist=smoothedMap[topic];
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
	
public ArrayList<ArrayList<String>> getMostProbableWordsArray(double theta,int limit){
		
		ArrayList<ArrayList<String>> result=new ArrayList<ArrayList<String>>();
		ArrayList<SmoothedUnit>[] smoothedMap=getSmoothedTauProb(theta);
		for(int topic=0;topic<numOfTopics;topic++){
			ArrayList<SmoothedUnit> subList=smoothedMap[topic];
			SmoothedUnit[] subArray=subList.toArray(new SmoothedUnit[subList.size()]);
			Arrays.sort(subArray);
			ArrayList<String> list=new ArrayList<String>();
			for(int i=subArray.length-1;i>=subArray.length-limit;i--){
				list.add(subArray[i].word);
			}
			result.add(list);
		}
		
		for(int topic=0;topic<numOfTopics;topic++){
			System.out.println("topic:"+topic);
			ArrayList<String> sublist=result.get(topic);
			for(String word:sublist){
				System.out.print(word+"\t");
			}
			System.out.println();
		}
		return result;
	}
	
	public void printMostProbableWords(double theta,int limit){
		ArrayList<ArrayList<String>> list=getMostProbableWords(theta,limit);
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
		Gibbs2 model=new Gibbs2(50,0.5,1000);
		model.parseTrainingFile(args[0]);
		System.out.println("File Parsed");
		model.trainParameters(0.01);
		model.printDeltaProb(16);
		model.printMostProbableWords(5,15);
		System.out.println("Finished");
	}
}





