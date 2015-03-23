gridRows = len(idxs)
gridColumns = 2

document = ve.Embedded("doc_1")
page = document.Root.Add('page')#, width = pageWidth, height=pageHeight)
grid = page.Add('grid', autoadd=False, rows = gridRows, columns = gridColumns,
									bottomMargin='2cm',
									leftMargin='2.5cm'
									)

xAxis = graph.Add('axis', name='x', label = self.xlabel,
						log = self.setxlog, 
						min = self.xAxisLimits[0],
						max = self.xAxisLimits[1],
						Label__size = '25pt',
						Label__font = self.font,
						TickLabels__size = '20pt',
						MajorTicks__width = '2pt',
						MajorTicks__length = '10pt',
						MinorTicks__width = '1pt',
						MinorTicks__length = '6pt',
					)
			yAxis = graph.Add('axis', name='y', label = self.ylabel, direction = 'vertical',
						log = self.setylog, 
						min = self.yAxisLimits[0],
						max = self.yAxisLimits[1],
						Label__size = '25pt',
						Label__font = self.font,
						TickLabels__size = '20pt',
						TickLabels__format = 'auto',
						MajorTicks__width = '2pt',
						MajorTicks__length = '10pt',
						MinorTicks__width = '1pt',
						MinorTicks__length = '6pt',
					)
			
			graph.Add('xy', key=prop+": "+str(value),
												marker = self.objMarkers[zIdx],
												MarkerFill__color = self.colors[zIdx],
												markerSize = '5pt', 
												errorStyle = 'barends',
												ErrorBarLine__width = '1pt',
												ErrorBarLine__color = self.colors[zIdx],
												#ErrorBarLine__hideVert = False,
												PlotLine__hide = True, 
												FillBelow__hide = True,
												))
			
			
			dataNameY = "Y"+prop+"_"+str(value)
					dataNameX = "X"+prop+"_"+str(value)
					self.document.SetData(dataNameX, x)
					self.document.SetData(dataNameY, y, negerr=negerr, poserr=poserr)
					xyPlots[-1].xData.val = dataNameX
					xyPlots[-1].yData.val = dataNameY
					
				
				zIdx += 1	
				
			plotKey = graph.Add('key', autoadd=False, 
						horzPosn = 'right',
						vertPosn = 'top',
						Text__font = self.font,
						Text__size = '15',
						#Border__width = '1.5pt'
						Border__hide = True
						)
				

	
	def save(self, outname):
		self.document.Root.page1.grid1.Action('zeroMargins')
		self.document.Save(self.outpath+outname+".vsz")
		self.document.Export(self.outpath+outname+".pdf")