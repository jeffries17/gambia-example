import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
  Divider,
  Button,
  Tabs,
  Tab,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  Download as DownloadIcon,
  Share as ShareIcon,
  Print as PrintIcon,
  GetApp as ExportIcon
} from '@mui/icons-material';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
import { saveAs } from 'file-saver';
import Papa from 'papaparse';

// Custom color palette
const COLORS = ['#00C49F', '#0088FE', '#FF8042', '#FFBB28', '#8884d8'];
const SENTIMENT_COLORS = {
  positive: '#4caf50',
  neutral: '#2196f3',
  negative: '#f44336'
};

const ResultsDisplay = ({ results, isLoading = false, error = null, jobId = null }) => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (isLoading) {
    return (
      <Card elevation={2} sx={{ p: 3, textAlign: 'center' }}>
        <CircularProgress sx={{ my: 3 }} />
        <Typography variant="h6" sx={{ mb: 1 }}>
          Processing Your Analysis
        </Typography>
        <Typography color="text.secondary" paragraph>
          We're analyzing your reviews. This may take a minute...
        </Typography>
        {jobId && (
          <Typography variant="caption" color="text.secondary">
            Job ID: {jobId}
          </Typography>
        )}
      </Card>
    );
  }

  if (error) {
    return (
      <Card elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" color="error" sx={{ mb: 1 }}>
          Analysis Error
        </Typography>
        <Typography paragraph>{error}</Typography>
        <Button variant="outlined" onClick={() => window.location.reload()}>
          Try Again
        </Button>
      </Card>
    );
  }

  if (!results) {
    return null;
  }

  // Extract data from results
  const overall = results.overall || {};
  const aspects = results.aspects || {};
  const topPhrases = results.top_phrases || [];

  // Prepare data for charts
  const sentimentData = [
    { name: 'Positive', value: overall.positive_percentage || 0, color: SENTIMENT_COLORS.positive },
    { name: 'Neutral', value: overall.neutral_percentage || 0, color: SENTIMENT_COLORS.neutral },
    { name: 'Negative', value: overall.negative_percentage || 0, color: SENTIMENT_COLORS.negative }
  ];

  const aspectData = Object.entries(aspects).map(([name, data], index) => ({
    name,
    sentiment: data.sentiment * 100, // Convert to percentage
    count: data.count,
    color: COLORS[index % COLORS.length]
  })).sort((a, b) => b.count - a.count).slice(0, 8); // Top 8 aspects by count

  // Export functions
  const exportToPDF = () => {
    const doc = new jsPDF();
    
    // Add title
    doc.setFontSize(20);
    doc.text('Sentiment Analysis Report', 20, 20);
    doc.setFontSize(12);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 30);
    
    // Overall stats
    doc.setFontSize(16);
    doc.text('Overall Results', 20, 45);
    doc.setFontSize(12);
    doc.text(`Reviews Analyzed: ${overall.review_count || 0}`, 20, 55);
    doc.text(`Average Sentiment: ${overall.average_sentiment?.toFixed(2) || 0}`, 20, 62);
    doc.text(`Positive: ${overall.positive_percentage || 0}%`, 20, 69);
    doc.text(`Neutral: ${overall.neutral_percentage || 0}%`, 20, 76);
    doc.text(`Negative: ${overall.negative_percentage || 0}%`, 20, 83);

    // Aspects
    doc.setFontSize(16);
    doc.text('Key Aspects', 20, 100);
    const aspectRows = Object.entries(aspects).map(([aspect, data]) => [
      aspect,
      data.sentiment.toFixed(2),
      data.count
    ]);
    
    doc.autoTable({
      startY: 105,
      head: [['Aspect', 'Sentiment', 'Mentions']],
      body: aspectRows,
    });
    
    // Top phrases
    const currentY = doc.lastAutoTable.finalY + 15;
    doc.setFontSize(16);
    doc.text('Top Phrases', 20, currentY);
    
    const phraseRows = topPhrases.map(phrase => [
      phrase.text,
      phrase.sentiment.toFixed(2),
      phrase.count
    ]);
    
    doc.autoTable({
      startY: currentY + 5,
      head: [['Phrase', 'Sentiment', 'Count']],
      body: phraseRows,
    });
    
    // Save the PDF
    doc.save('sentiment-analysis-report.pdf');
  };

  const exportToCSV = () => {
    // Prepare dataset
    const csvData = [
      ['Overall Statistics'],
      ['Reviews Analyzed', overall.review_count || 0],
      ['Average Sentiment', overall.average_sentiment || 0],
      ['Positive Percentage', `${overall.positive_percentage || 0}%`],
      ['Neutral Percentage', `${overall.neutral_percentage || 0}%`],
      ['Negative Percentage', `${overall.negative_percentage || 0}%`],
      [],
      ['Aspect Analysis'],
      ['Aspect', 'Sentiment', 'Mentions']
    ];
    
    // Add aspect data
    Object.entries(aspects).forEach(([aspect, data]) => {
      csvData.push([aspect, data.sentiment.toFixed(2), data.count]);
    });
    
    csvData.push([]);
    csvData.push(['Top Phrases']);
    csvData.push(['Phrase', 'Sentiment', 'Count']);
    
    // Add phrase data
    topPhrases.forEach(phrase => {
      csvData.push([phrase.text, phrase.sentiment.toFixed(2), phrase.count]);
    });
    
    // Convert to CSV
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'sentiment-analysis-data.csv');
  };

  // Render content based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // Overview
        return (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Overall Sentiment
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={sentimentData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}%`}
                        >
                          {sentimentData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <RechartsTooltip formatter={(value) => `${value}%`} />
                      </PieChart>
                    </ResponsiveContainer>
                  </Box>
                  <Box sx={{ mt: 2, textAlign: 'center' }}>
                    <Typography variant="h3" fontWeight="bold" color="primary">
                      {overall.average_sentiment ? (overall.average_sentiment * 100).toFixed(0) : 0}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Overall Positive Sentiment
                    </Typography>
                  </Box>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={8}>
                <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Key Metrics
                  </Typography>
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" fontWeight="bold">
                          {overall.review_count || 0}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Reviews Analyzed
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" fontWeight="bold" color={SENTIMENT_COLORS.positive}>
                          {overall.positive_percentage || 0}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Positive
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" fontWeight="bold" color={SENTIMENT_COLORS.neutral}>
                          {overall.neutral_percentage || 0}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Neutral
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" fontWeight="bold" color={SENTIMENT_COLORS.negative}>
                          {overall.negative_percentage || 0}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Negative
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                  
                  <Typography variant="subtitle1" sx={{ mt: 3, mb: 1 }}>
                    Top Positive Phrases
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {topPhrases
                      .filter(phrase => phrase.sentiment > 0)
                      .slice(0, 5)
                      .map((phrase, index) => (
                        <Chip 
                          key={index}
                          label={`${phrase.text} (${phrase.count})`}
                          color="success"
                          size="small"
                          variant="outlined"
                        />
                      ))}
                  </Box>
                  
                  <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                    Top Negative Phrases
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {topPhrases
                      .filter(phrase => phrase.sentiment < 0)
                      .slice(0, 5)
                      .map((phrase, index) => (
                        <Chip 
                          key={index}
                          label={`${phrase.text} (${phrase.count})`}
                          color="error"
                          size="small"
                          variant="outlined"
                        />
                      ))}
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        );
        
      case 1: // Aspects
        return (
          <Box>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Paper elevation={2} sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Aspect Sentiment
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={aspectData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} />
                      <RechartsTooltip formatter={(value) => `${value.toFixed(0)}%`} />
                      <Legend />
                      <Bar dataKey="sentiment" name="Sentiment Score (%)" radius={[5, 5, 0, 0]}>
                        {aspectData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    Aspect Mentions
                  </Typography>
                  <List>
                    {aspectData.map((aspect, index) => (
                      <React.Fragment key={index}>
                        {index > 0 && <Divider />}
                        <ListItem>
                          <ListItemText 
                            primary={aspect.name} 
                            secondary={`${aspect.sentiment.toFixed(0)}% positive, ${aspect.count} mentions`}
                          />
                          <Chip 
                            label={aspect.count} 
                            size="small" 
                            color={aspect.sentiment > 70 ? "success" : aspect.sentiment < 40 ? "error" : "primary"}
                          />
                        </ListItem>
                      </React.Fragment>
                    ))}
                  </List>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        );
        
      case 2: // Phrases
        return (
          <Box>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Key Phrases & Themes
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Positive Mentions
                  </Typography>
                  <List>
                    {topPhrases
                      .filter(phrase => phrase.sentiment > 0)
                      .slice(0, 8)
                      .map((phrase, index) => (
                        <React.Fragment key={index}>
                          {index > 0 && <Divider />}
                          <ListItem>
                            <ListItemText 
                              primary={phrase.text}
                              secondary={`Mentioned ${phrase.count} times`}
                            />
                            <Chip 
                              label={`${(phrase.sentiment * 100).toFixed(0)}%`} 
                              color="success"
                              size="small"
                            />
                          </ListItem>
                        </React.Fragment>
                      ))}
                  </List>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold' }}>
                    Negative Mentions
                  </Typography>
                  <List>
                    {topPhrases
                      .filter(phrase => phrase.sentiment < 0)
                      .slice(0, 8)
                      .map((phrase, index) => (
                        <React.Fragment key={index}>
                          {index > 0 && <Divider />}
                          <ListItem>
                            <ListItemText 
                              primary={phrase.text}
                              secondary={`Mentioned ${phrase.count} times`}
                            />
                            <Chip 
                              label={`${(phrase.sentiment * 100).toFixed(0)}%`} 
                              color="error"
                              size="small"
                            />
                          </ListItem>
                        </React.Fragment>
                      ))}
                  </List>
                </Grid>
              </Grid>
            </Paper>
          </Box>
        );
        
      default:
        return null;
    }
  };

  return (
    <Card elevation={2}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="h5" fontWeight="bold" gutterBottom>
              Analysis Results
            </Typography>
            <Typography color="text.secondary">
              {overall.review_count || 0} reviews analyzed • {new Date().toLocaleDateString()}
            </Typography>
          </Box>
          
          <Box>
            <Tooltip title="Export PDF">
              <IconButton onClick={exportToPDF}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export CSV">
              <IconButton onClick={exportToCSV}>
                <ExportIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Print">
              <IconButton onClick={() => window.print()}>
                <PrintIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
          <Tab label="Overview" />
          <Tab label="Aspects" />
          <Tab label="Key Phrases" />
        </Tabs>
        
        {renderTabContent()}
      </CardContent>
    </Card>
  );
};

export default ResultsDisplay; 